import argparse
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from models.LBM_Conv_distill import UltraLight_VM_UNet
from dataset.dataset import NPY_datasets
from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

from torch.utils.tensorboard import SummaryWriter
from model_stats import analyze_model

import warnings
warnings.filterwarnings("ignore")


class HybridDistillationLoss(nn.Module):
    
    def __init__(self, 
                 hard_loss_weight=1.0,
                 soft_loss_weight=0.5, 
                 gradient_loss_weight=0.3,
                 attention_loss_weight=0.2,
                 temperature=3.0,
                 attention_temperature=0.5,
                 criterion=None):
        super(HybridDistillationLoss, self).__init__()
        
        # Loss weights
        self.hard_loss_weight = hard_loss_weight
        self.soft_loss_weight = soft_loss_weight
        self.gradient_loss_weight = gradient_loss_weight
        self.attention_loss_weight = attention_loss_weight
        
        self.temperature = temperature
        self.attention_temperature = attention_temperature
        
        self.criterion = criterion

        
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def _decoupled_kl_loss(self, student_prob, teacher_prob):
        eps = 1e-6
        
        student_prob = torch.clamp(student_prob, eps, 1.0 - eps)
        teacher_prob = torch.clamp(teacher_prob, eps, 1.0 - eps).detach()
        
        kl_pos = teacher_prob * (torch.log(teacher_prob) - torch.log(student_prob))
        kl_neg = (1 - teacher_prob) * (torch.log(1 - teacher_prob) - torch.log(1 - student_prob))
        
        kl_loss = kl_pos + kl_neg
        kl_loss = torch.where(torch.isnan(kl_loss) | torch.isinf(kl_loss), 
                             torch.zeros_like(kl_loss), 
                             kl_loss)
        
        return 8.0 * kl_loss.mean()
    
    def _gradient_matching_loss(self, student_prob, teacher_prob):
        student_grad_x = F.conv2d(student_prob, self.sobel_x, padding=1)
        student_grad_y = F.conv2d(student_prob, self.sobel_y, padding=1)
        
        teacher_grad_x = F.conv2d(teacher_prob.detach(), self.sobel_x, padding=1)
        teacher_grad_y = F.conv2d(teacher_prob.detach(), self.sobel_y, padding=1)
        
        student_grad_mag = torch.sqrt(student_grad_x**2 + student_grad_y**2 + 1e-8)
        teacher_grad_mag = torch.sqrt(teacher_grad_x**2 + teacher_grad_y**2 + 1e-8)
        
        grad_loss = F.mse_loss(student_grad_mag, teacher_grad_mag)
        
        return grad_loss
    
    def _attention_transfer_loss(self, student_prob, teacher_prob):
        eps = 1e-6
        
        B = student_prob.size(0)
        student_flat = student_prob.view(B, -1)
        teacher_flat = teacher_prob.view(B, -1).detach()
        
        student_attention = F.softmax(student_flat / self.attention_temperature, dim=1)
        teacher_attention = F.softmax(teacher_flat / self.attention_temperature, dim=1)
        
        attention_loss = F.kl_div(
            student_attention.log(),
            teacher_attention,
            reduction='batchmean'
        )
        
        if torch.isnan(attention_loss) or torch.isinf(attention_loss):
            return torch.tensor(0.0, device=student_prob.device)
        
        return attention_loss

    def forward(self, student_outputs, teacher_outputs, targets):

        if isinstance(student_outputs, (tuple, list)):
            student_deep_sup, student_final = student_outputs
        else:
            student_deep_sup = None
            student_final = student_outputs
        
        if isinstance(teacher_outputs, (tuple, list)):
            teacher_deep_sup, teacher_final = teacher_outputs
        else:
            teacher_deep_sup = None
            teacher_final = teacher_outputs
        
        if student_deep_sup is not None and isinstance(student_deep_sup, (tuple, list)):
            hard_loss = 0
            for s_out in student_deep_sup:
                hard_loss += self.criterion(s_out, targets)
            hard_loss += self.criterion(student_final, targets)
            hard_loss = hard_loss / (len(student_deep_sup) + 1)
        else:
            hard_loss = self.criterion(student_final, targets)

        soft_loss = self._decoupled_kl_loss(student_final, teacher_final)
        
        if (student_deep_sup is not None and teacher_deep_sup is not None and
            isinstance(student_deep_sup, (tuple, list)) and isinstance(teacher_deep_sup, (tuple, list))):
            for s_out, t_out in zip(student_deep_sup, teacher_deep_sup):
                soft_loss += self._decoupled_kl_loss(s_out, t_out)
            soft_loss = soft_loss / (len(student_deep_sup) + 1)
        
        gradient_loss = self._gradient_matching_loss(student_final, teacher_final)
        
        attention_loss = self._attention_transfer_loss(student_final, teacher_final)
        
        hard_loss = torch.where(torch.isnan(hard_loss), 
                               torch.tensor(0.0, device=hard_loss.device), 
                               hard_loss)
        soft_loss = torch.where(torch.isnan(soft_loss), 
                               torch.tensor(0.0, device=soft_loss.device), 
                               soft_loss)
        gradient_loss = torch.where(torch.isnan(gradient_loss), 
                                   torch.tensor(0.0, device=gradient_loss.device), 
                                   gradient_loss)
        attention_loss = torch.where(torch.isnan(attention_loss), 
                                    torch.tensor(0.0, device=attention_loss.device), 
                                    attention_loss)
        
        total_loss = (
            self.hard_loss_weight * hard_loss +
            self.soft_loss_weight * soft_loss +
            self.gradient_loss_weight * gradient_loss +
            self.attention_loss_weight * attention_loss
        )

        
        loss_dict = {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss.item() if isinstance(hard_loss, torch.Tensor) else hard_loss,
            'soft_loss': soft_loss.item(),
            'gradient_loss': gradient_loss.item(),
            'attention_loss': attention_loss.item(),
            'attention_loss_full': f"{attention_loss.item():.15e}",
        }
        
        return total_loss, loss_dict


def load_teacher_model(teacher_path, device='cuda'):
    """
    Load pre-trained teacher model with flexible loading
    
    Args:
        teacher_path: path to teacher model checkpoint
        device: device to load model on
    
    Returns:
        teacher_model: loaded and frozen teacher model
    """
    print(f'#----------Loading Teacher Model from {teacher_path}----------#')
    
    teacher_model = UltraLight_VM_UNet(channel_multiplier=1.0)
    
    if os.path.exists(teacher_path):
        checkpoint = torch.load(teacher_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        try:
            teacher_model.load_state_dict(state_dict, strict=True)
            print(f'Successfully loaded teacher model from {teacher_path} (strict mode)')
        except RuntimeError as e:
            print('Strict loading failed. Attempting flexible loading...')
            print(f'   Error: {str(e)[:200]}...')
            
            missing_keys, unexpected_keys = teacher_model.load_state_dict(
                state_dict, strict=False
            )

            total_keys = len(teacher_model.state_dict())
            loaded_keys = total_keys - len(missing_keys)
            load_ratio = loaded_keys / total_keys
            
            print(f'   Missing keys: {len(missing_keys)}')
            print(f'   Unexpected keys: {len(unexpected_keys)}')
            print(f'   Loaded: {loaded_keys}/{total_keys} parameters ({load_ratio*100:.1f}%)')
            
    else:
        raise FileNotFoundError(f'Teacher model not found at {teacher_path}')
    
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    teacher_model = teacher_model.float()

    for param in teacher_model.parameters():
        param.requires_grad = False
    
    return teacher_model


def train_one_epoch_distillation(train_loader, student_model, teacher_model, 
                                 distillation_criterion, optimizer, scheduler, 
                                 epoch, logger, config, scaler=None):
    student_model.train()
    teacher_model.eval()
    
    if hasattr(distillation_criterion, 'set_epoch'):
        distillation_criterion.set_epoch(epoch)
    
    epoch_loss = 0
    epoch_loss_components = {
        'hard_loss': 0,
        'soft_loss': 0, 
        'gradient_loss': 0,
        'attention_loss': 0
    }
    
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='batch') as pbar:
        for iter_num, data in enumerate(train_loader):
            optimizer.zero_grad()
            
            images, targets = data
            images = images.float().cuda()
            targets = targets.float().cuda()
            
            if scaler is not None and config.amp:
                with autocast():
                    with torch.no_grad():
                        teacher_outputs = teacher_model(images)
                    
                    student_outputs = student_model(images)
                
                loss, loss_dict = distillation_criterion(
                    student_outputs, teacher_outputs, targets
                )
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                
                student_outputs = student_model(images)
                loss, loss_dict = distillation_criterion(
                    student_outputs, teacher_outputs, targets
                )
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            for key in epoch_loss_components.keys():
                if key in loss_dict:
                    epoch_loss_components[key] += loss_dict[key]
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'hard': f'{loss_dict.get("hard_loss", 0):.4f}',
                'soft': f'{loss_dict.get("soft_loss", 0):.4f}',
                'grad': f'{loss_dict.get("gradient_loss", 0):.4f}',
                'bdry': f'{loss_dict.get("boundary_loss", 0):.4f}',
                'attn': f'{loss_dict.get("attention_loss", 0):.4f}',
                'attn_full': loss_dict.get("attention_loss_full", "0.0e+00"),
            })
            pbar.update(1)
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    for key in epoch_loss_components.keys():
        epoch_loss_components[key] /= len(train_loader)
    
    if scheduler is not None:
        scheduler.step()
    
    logger.info(f'Epoch {epoch} Training Loss: {avg_epoch_loss:.4f}')
    logger.info(f'  Hard Loss: {epoch_loss_components["hard_loss"]:.4f}')
    logger.info(f'  Soft Loss (DKD): {epoch_loss_components["soft_loss"]:.4f}')
    logger.info(f'  Gradient Loss: {epoch_loss_components["gradient_loss"]:.4f}')
    logger.info(f'  Boundary Loss: {epoch_loss_components["boundary_loss"]:.4f}')
    logger.info(f'  Attention Loss: {epoch_loss_components["attention_loss"]:.4f}')
    
    return avg_epoch_loss, epoch_loss_components


def main(config, teacher_model_path, seed=None):
    
    if seed is not None:
        config.seed = seed
    
    original_work_dir = config.work_dir

    config.work_dir = original_work_dir.rstrip('/') + f'_distillation_seed{config.seed}/'
    
    writer = SummaryWriter(os.path.join(config.work_dir, 'tensorboard'))
    
    set_seed(config.seed)
    
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)
    
    log_dir = os.path.join(config.work_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger('distillation_train', log_dir)
    
    print(f'Work directory: {config.work_dir}')
    print(f'Random seed: {config.seed}')
    print(f'Dataset: {config.datasets}')
    
    logger.info('='*70)
    logger.info('BOUNDARY-FOCUSED KNOWLEDGE DISTILLATION TRAINING')
    logger.info('='*70)
    logger.info(f'Teacher model: {teacher_model_path}')
    logger.info(f'Work directory: {config.work_dir}')
    logger.info(f'Random seed: {config.seed}')
    
    print('#----------GPU init----------#')
    gpu_ids = [0]
    torch.cuda.empty_cache()
    
    print('#----------Preparing datasets----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False
    )
    
    logger.info(f'Training dataset size: {len(train_dataset)}')
    logger.info(f'Validation dataset size: {len(val_dataset)}')
    
    print('#----------Loading Teacher Model----------#')
    teacher_model = load_teacher_model(teacher_model_path, device='cuda')
    teacher_model = torch.nn.DataParallel(
        teacher_model, device_ids=gpu_ids, output_device=gpu_ids[0]
    )
    
    print('#----------Teacher Model Statistics----------#')
    try:
        if hasattr(config, 'input_size'):
            input_size = config.input_size
        else:
            input_size = (3, 256, 256)
        
        teacher_stats = analyze_model(
            model=teacher_model, 
            input_size=input_size, 
            device='cuda:0',
            batch_sizes=[1]
        )
        
        teacher_params = teacher_stats.count_parameters()
        logger.info(f"Teacher Model Parameters: {teacher_params['total']:,} "
                   f"({teacher_params['total']/1e6:.2f}M)")
        
    except Exception as e:
        logger.warning(f"Teacher model analysis failed: {e}")

    print('#----------Creating Student Model----------#')
  
    student_model = UltraLight_VM_UNet(channel_multiplier=0.5)

    
    student_model = torch.nn.DataParallel(
        student_model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0]
    )
    
    student_model = student_model.float()
    logger.info("Student model set to float32")
    
    print('#----------Student Model Statistics----------#')
    try:
        student_stats = analyze_model(
            model=student_model, 
            input_size=input_size, 
            device='cuda:0',
            batch_sizes=[1, config.batch_size]
        )
        
        student_params = student_stats.count_parameters()
        logger.info(f"Student Model Parameters: {student_params['total']:,} "
                   f"({student_params['total']/1e6:.2f}M)")
        
        compression_ratio = teacher_params['total'] / student_params['total']
        logger.info(f"Compression Ratio: {compression_ratio:.2f}x")
        gflops = student_stats.calculate_gflops()
        if gflops:
            logger.info(f"Model GFLOPs: {gflops:.2f}G")

        stats_file = os.path.join(config.work_dir, 'distillation_model_analysis.json')
        stats_dict = {
            'teacher': teacher_stats.count_parameters(),
            'student': student_stats.count_parameters(),
            'compression_ratio': compression_ratio,
            'student_gflops': student_stats.calculate_gflops()
        }
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=4)
        logger.info(f"Model comparison saved to: {stats_file}")
        
    except Exception as e:
        logger.warning(f"Student model analysis failed: {e}")
    

    distillation_criterion = HybridDistillationLoss(
    hard_loss_weight=1.0,
    soft_loss_weight=0.5,
    gradient_loss_weight=0.3,
    attention_loss_weight=0.2,
    criterion=config.criterion  # BceDiceLoss
    ).cuda()
    
    logger.info('Distillation loss components:')
    logger.info('  Hard loss weight: 1.0 (Unified Focal)')
    logger.info('  Soft loss weight: 0.5 (Decoupled KD)')
    logger.info('  Gradient loss weight: 0.3 (Edge Matching)')
    logger.info('  Boundary loss weight: 0.4 (Boundary DoU, with scheduling)')
    logger.info('  Attention loss weight: 0.2 (Spatial Focus Transfer)')
    logger.info('  Temperature: 3.0')

    
    scaler = None
    optimizer = get_optimizer(config, student_model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()
    print('#----------Starting Distillation Training----------#')
    best_loss = float('inf')
    
    for epoch in range(1, config.epochs + 1):
        train_loss, loss_components = train_one_epoch_distillation(
            train_loader,
            student_model,
            teacher_model,
            distillation_criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler
        )
        

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/hard', loss_components['hard_loss'], epoch)
        writer.add_scalar('Loss/soft', loss_components['soft_loss'], epoch)
        writer.add_scalar('Loss/gradient', loss_components['gradient_loss'], epoch)
        writer.add_scalar('Loss/boundary', loss_components['boundary_loss'], epoch)
        writer.add_scalar('Loss/attention', loss_components['attention_loss'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(
                student_model.module.state_dict(),
                os.path.join(checkpoint_dir, 'best.pth')
            )
            logger.info(f'✓ Best model saved at epoch {epoch} with loss {best_loss:.4f}')
        
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': student_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
                'min_loss': best_loss
            },
            os.path.join(checkpoint_dir, 'latest.pth')
        )
    
    print('#----------Testing Distilled Student Model----------#')
    best_weight = torch.load(
        os.path.join(checkpoint_dir, 'best.pth'),
        map_location=torch.device('cpu')
    )
    student_model.module.load_state_dict(best_weight)
    
    loss = test_one_epoch(
        val_loader,
        student_model,
        config.criterion,
        logger,
        config,
    )
    
    final_model_name = f'best-distilled-epoch{config.epochs}-loss{loss:.4f}.pth'
    os.rename(
        os.path.join(checkpoint_dir, 'best.pth'),
        os.path.join(checkpoint_dir, final_model_name)
    )
    
    logger.info(f'Training completed! Best model: {final_model_name}')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Boundary-Focused Knowledge Distillation Training')
    
    parser.add_argument('--teacher_path', type=str, required=True,
                       help='Path to pre-trained teacher model checkpoint')
    
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (optional, uses config seed if not provided)')
    
    args = parser.parse_args()
    
    config = setting_config
    
    print('Starting Boundary-Focused Knowledge Distillation Training')
    print(f'Teacher model: {args.teacher_path}')
    print(f'Seed: {args.seed if args.seed else f"{config.seed} (from config)"}')
    print('')
    
    main(config, args.teacher_path, args.seed)
