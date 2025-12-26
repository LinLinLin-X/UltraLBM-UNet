import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba


class DepthwiseSeparableConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, act_layer=nn.GELU):
        super().__init__()
        # Depthwise Conv
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.act_dw = act_layer()

        # Pointwise Conv
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.act_pw = act_layer()

    def forward(self, x):
        x = self.dw(x)
        x = self.bn_dw(x)
        x = self.act_dw(x)

        x = self.pw(x)
        x = self.bn_pw(x)
        x = self.act_pw(x)
        return x



class LMBP(nn.Module):

    def __init__(self, input_dim, output_dim, sep_conv_kernel=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert input_dim % 4 == 0
        self.norm = nn.LayerNorm(input_dim)
        self.sep_conv_kernel = sep_conv_kernel
        
        padding = (self.sep_conv_kernel - 1) // 2 
        c_quarter = input_dim // 4

        self.sep_conv1 = DepthwiseSeparableConvBNAct(
            in_ch=c_quarter,
            out_ch=c_quarter,
            kernel_size=sep_conv_kernel,
            stride=1,
            padding=padding,
            act_layer=nn.GELU,
        )
        self.sep_conv2 = DepthwiseSeparableConvBNAct(
            in_ch=c_quarter,
            out_ch=c_quarter,
            kernel_size=sep_conv_kernel,
            stride=1,
            padding=padding,
            act_layer=nn.GELU,
        )
        self.sep_conv3 = DepthwiseSeparableConvBNAct(
            in_ch=c_quarter,
            out_ch=c_quarter,
            kernel_size=sep_conv_kernel,
            stride=1,
            padding=padding,
            act_layer=nn.GELU,
        )
        
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        c_quarter = C // 4
        
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2) # (B, N, C)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2) # (B, N, C/4) each

        x1_img = x1.transpose(-1, -2).reshape(B, c_quarter, *img_dims) # (B, C/4, H, W)
        x1_img = self.sep_conv1(x1_img)
        x1_flat = x1_img.reshape(B, c_quarter, n_tokens).transpose(-1, -2) # (B, N, C/4)
        x_proc1 = x1_flat + self.skip_scale * x1

        x2_img = x2.transpose(-1, -2).reshape(B, c_quarter, *img_dims)
        x2_img = self.sep_conv2(x2_img)
        x2_flat = x2_img.reshape(B, c_quarter, n_tokens).transpose(-1, -2)
        x_proc2 = x2_flat + self.skip_scale * x2
        
        x3_img = x3.transpose(-1, -2).reshape(B, c_quarter, *img_dims)
        x3_img = self.sep_conv3(x3_img)
        x3_flat = x3_img.reshape(B, c_quarter, n_tokens).transpose(-1, -2)
        x_proc3 = x3_flat + self.skip_scale * x3
        
        x_proc4 = x4 + self.skip_scale * x4

        x_mamba = torch.cat([x_proc1, x_proc2, x_proc3, x_proc4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class GLMBP(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, sep_conv_kernel=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert input_dim % 4 == 0
        self.norm = nn.LayerNorm(input_dim)
        self.sep_conv_kernel = sep_conv_kernel
        
        self.mamba = Mamba(
            d_model=input_dim // 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        padding = (self.sep_conv_kernel - 1) // 2 

        self.sep_conv = DepthwiseSeparableConvBNAct(
            in_ch=input_dim // 4,
            out_ch=input_dim // 4,
            kernel_size=self.sep_conv_kernel,
            stride=1,
            padding=padding,
            act_layer=nn.GELU,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)

        x1_backward = torch.flip(x1, dims=[2])
        x_mamba1_forward = self.mamba(x1)
        x_mamba1_backward = torch.flip(self.mamba(x1_backward), dims=[2])
        x_mamba1 = x_mamba1_forward + x_mamba1_backward + self.skip_scale * x1


        x_mamba2_forward = self.mamba(x2)
        x2_backward = torch.flip(x2, dims=[2])
        x_mamba2_backward = torch.flip(self.mamba(x2_backward), dims=[2])
        x_mamba2 = x_mamba2_forward + x_mamba2_backward + self.skip_scale * x2
        
        c_quarter = C // 4
        x3_img = x3.transpose(-1, -2).reshape(B, c_quarter, *img_dims)
        x3_img = self.sep_conv(x3_img)
        x3_flat = x3_img.reshape(B, c_quarter, n_tokens).transpose(-1, -2)
        x_mamba3 = x3_flat + self.skip_scale * x3
        
        x_mamba4 = x4 + self.skip_scale * x4

        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

class UltraLBM_UNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, 
                 c_list=None,
                 channel_multiplier=1.0,
                 init_k=1, learnable_skip=True):
        super().__init__()

        if c_list is None:
            base_c_list = [8, 16, 24, 32, 48, 64]
            c_list = [int(c * channel_multiplier) for c in base_c_list]
            c_list = [max(4, (c // 4) * 4) for c in c_list]
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            LMBP(input_dim=c_list[2], output_dim=c_list[3], sep_conv_kernel=3)
        )
        self.encoder5 = nn.Sequential(
            GLMBP(input_dim=c_list[3], output_dim=c_list[4], sep_conv_kernel=5)
        )
        self.encoder6 = nn.Sequential(
            GLMBP(input_dim=c_list[4], output_dim=c_list[5], sep_conv_kernel=7)
        )
        self.init_k = float(init_k)
        
        self.decoder1 = nn.Sequential(
            GLMBP(input_dim=c_list[5], output_dim=c_list[4], sep_conv_kernel=7)
        ) 
        self.decoder2 = nn.Sequential(
            GLMBP(input_dim=c_list[4], output_dim=c_list[3], sep_conv_kernel=5)
        ) 
        self.decoder3 = nn.Sequential(
            LMBP(input_dim=c_list[3], output_dim=c_list[2], sep_conv_kernel=3)
        ) 
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        ) 
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        ) 

        self.ebn1 = nn.GroupNorm(min(4, c_list[0]), c_list[0])
        self.ebn2 = nn.GroupNorm(min(4, c_list[1]), c_list[1])
        self.ebn3 = nn.GroupNorm(min(4, c_list[2]), c_list[2])
        self.ebn4 = nn.GroupNorm(min(4, c_list[3]), c_list[3])
        self.ebn5 = nn.GroupNorm(min(4, c_list[4]), c_list[4])
        self.dbn1 = nn.GroupNorm(min(4, c_list[4]), c_list[4])
        self.dbn2 = nn.GroupNorm(min(4, c_list[3]), c_list[3])
        self.dbn3 = nn.GroupNorm(min(4, c_list[2]), c_list[2])
        self.dbn4 = nn.GroupNorm(min(4, c_list[1]), c_list[1])
        self.dbn5 = nn.GroupNorm(min(4, c_list[0]), c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)
        self.learnable_skip = learnable_skip
        if self.learnable_skip:
            self.k = nn.Parameter(torch.tensor(float(init_k)))
        else:
            self.register_buffer("skip_gammas", None)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32


        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32
        
        if self.learnable_skip:
            out5 = F.gelu(self.dbn1(self.decoder1(out)))
            out5 = out5 + self.k * t5

            out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),
                                        scale_factor=(2, 2),
                                        mode='bilinear',
                                        align_corners=True))
            out4 = out4 + self.k * t4

            out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),
                                        scale_factor=(2, 2),
                                        mode='bilinear',
                                        align_corners=True))
            out3 = out3 + self.k * t3

            out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),
                                        scale_factor=(2, 2),
                                        mode='bilinear',
                                        align_corners=True))
            out2 = out2 + self.k * t2

            out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),
                                        scale_factor=(2, 2),
                                        mode='bilinear',
                                        align_corners=True))
            out1 = out1 + self.k * t1
        else:
            out5 = F.gelu(self.dbn1(self.decoder1(out)))
            out5 = torch.add(out5, t5)

            out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), 
                                        scale_factor=(2, 2), 
                                        mode='bilinear', 
                                        align_corners=True))
            out4 = torch.add(out4, t4)
            
            out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), 
                                        scale_factor=(2, 2), 
                                        mode='bilinear', 
                                        align_corners=True))
            out3 = torch.add(out3, t3)
            
            out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), 
                                        scale_factor=(2, 2), 
                                        mode='bilinear', 
                                        align_corners=True))
            out2 = torch.add(out2, t2)
            
            out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), 
                                        scale_factor=(2, 2), 
                                        mode='bilinear', 
                                        align_corners=True))
            out1 = torch.add(out1, t1)
            
        out0 = F.interpolate(self.final(out1), 
                            scale_factor=(2, 2), 
                            mode='bilinear', 
                            align_corners=True)
        
        return torch.sigmoid(out0)
    
if __name__ == "__main__":
    model = UltraLBM_UNet()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
