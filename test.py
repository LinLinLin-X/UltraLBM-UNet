import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import sys
import warnings
from utils import * 
from dataset.dataset import NPY_datasets
from configs.config_setting import setting_config
import glob
from tqdm import tqdm
import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix
import argparse

from models.UltraLBM_UNet import UltraLBM_UNet
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

global logger

def test_model(config, work_dir, seed):
    
    print('#----------Creating logger and directories----------#')
    sys.path.append(work_dir + '/')
    log_dir = os.path.join(work_dir, 'log/isic17')
    checkpoint_dir = os.path.join(work_dir, 'checkpoints')
    best_weight_files = glob.glob(os.path.join(checkpoint_dir, '*best*.pth'))
    best_weight_path = best_weight_files[0]
    print(f"Best weight file: {best_weight_path}")
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
        return
    
    test_output_dir = os.path.join(work_dir, 'isic17_result')
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
        print(f"Created dedicated test output directory: {test_output_dir}")

    global logger
    logger = get_logger('test', log_dir)
    log_config_info(config, logger)

    
    print('#----------GPU init----------#')
    set_seed(seed)
    gpu_ids = [0]
    torch.cuda.empty_cache()

    print('#----------Preparing Test Dataset----------#')
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1, 
                                shuffle=False, 
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    
    if not os.path.exists(best_weight_path):
        logger.error(f"Best weight file not found at: {best_weight_path}")
        print(f"Error: Best weight file not found at: {best_weight_path}")
        return


    print('#----------Prepareing Models and Loading Weights----------#')
    model = UltraLBM_UNet()
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
    
    try:
        best_weight = torch.load(best_weight_path, map_location=torch.device('cpu'))
        model.module.load_state_dict(best_weight) 
        print(f"Successfully loaded best weight from {best_weight_path}")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        print(f"Failed to load model weights: {e}")
        return
    
    criterion = config.criterion
    
    
    print('#----------Starting Testing and Saving Results----------#')
    
    test_loss = test_one_epoch_ph2(
                    val_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_output_dir 
                )
    
    logger.info(f"Final Test Loss (Validation Loss): {test_loss:.4f}")
    print(f"Testing Complete. Final Loss: {test_loss:.4f}. Results saved to: {test_output_dir}")

def test_one_epoch_ph2(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   work_dir,
                   test_data_name=None):
    
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    save_dir = work_dir 

    os.makedirs(save_dir + '/original/', exist_ok=True)
    os.makedirs(save_dir + '/groundtruth/', exist_ok=True)
    os.makedirs(save_dir + '/prediction/', exist_ok=True)
    
    logger.info(f"Saving test results to: {save_dir}")

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            msk_display = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk_display)

            if type(out) is tuple:
                out = out[0]
            out_display = out.squeeze(1).cpu().detach().numpy()
            preds.append(out_display)

            if i % 1 == 0:
                prefix = f"{test_data_name}_{i}" if test_data_name else f"{i}"

                img_np = img[0].cpu().numpy()
                if img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                img_min = img_np.min()
                img_max = img_np.max()
                if img_max > img_min:
                    img_np = (img_np - img_min) / (img_max - img_min) * 255
                else:
                    img_np = np.zeros_like(img_np) * 255
                    
                img_np = img_np.astype(np.uint8)
                Image.fromarray(img_np).save(os.path.join(save_dir, 'original', f"{prefix}.png"))

                msk_np = msk_display[0]
                msk_binary = (msk_np >= 0.5).astype(np.uint8) * 255
                Image.fromarray(msk_binary).save(os.path.join(save_dir, 'groundtruth', f"{prefix}.png"))

                out_np = out_display[0]
                out_binary = (out_np >= config.threshold).astype(np.uint8) * 255
                Image.fromarray(out_binary).save(os.path.join(save_dir, 'prediction', f"{prefix}.png"))
                
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info_name = f'Test_datasets_name: {test_data_name}'
            print(log_info_name)
            logger.info(log_info_name)
            
        log_info_metrics = f'Test_metrics - Loss: {np.mean(loss_list):.4f}, mIoU: {miou:.4f}, DSC/F1: {f1_or_dsc:.4f}, Accuracy: {accuracy:.4f}, Specificity: {specificity:.4f}, Sensitivity: {sensitivity:.4f}, Confusion_Matrix:\n{confusion}'
        print(log_info_metrics)
        logger.info(log_info_metrics)
        
    return np.mean(loss_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PH2 External Test Script')
    
    parser.add_argument(
        '--work_dir', 
        type=str, 
        required=True,
        help='The root directory of the trained model (e.g., /path/to/isic_run/)'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()
    
    config = setting_config
    
    test_model(config, args.work_dir, args.seed)
    
    print("Test script finished.")
