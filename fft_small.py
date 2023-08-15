'''NIFF implementation for small, leight-weight models'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''
    NIFF MLP receiving two input feature maps and outputing planes output feature maps
    
    Args:
        planes (int): Desired number of output channels/multiplication weights.
        act (str): (Previously) option to change the acitvation within the MLP. Will be removed.
    ''' 
    
    def __init__(self, planes, act='silu'): 
        super(MLP, self).__init__()
        self.layer_mpl1 = nn.Conv2d(2, 8, 1, padding=0, groups=1)
        self.layer_mpl2 = nn.Conv2d(8, 4, 1, padding=0, groups=1)
        self.layer_mpl3 = nn.Conv2d(4, planes, 1, padding=0, groups=1)
        # self.act = act
        
    def forward(self, x):
        x = self.layer_mpl1(x.unsqueeze(0))
        x = F.silu(x)
        x = self.layer_mpl2(x)
        x = F.silu(x)
        x = self.layer_mpl3(x)
        return x

    
class FreqConv_DW_fft(nn.Module):
    '''
    Depthwise convolution inlcuding only the tranformation into the frequency domain via FFT.
    
    Args:
        planes (int): Number of input channels.
        imageheight (int): Feature map height.
        imagewidth (int): Feature map width.
    '''
    
    def __init__(self, planes, imageheight, imagewidth, act='relu', device='cuda'): 
        super(FreqConv_DW_fft, self).__init__()
        self.imageheight  = imageheight
        self.imagewidth = imagewidth
        self.planes = planes
        self.device = device
        self.mlp_imag = MLP(self.planes, act)
        self.mlp_real = MLP(self.planes, act)
        self.mask = torch.cat([
            torch.arange(-(self.imageheight/2), (self.imageheight/2), requires_grad=True)[None, :].repeat(self.imagewidth, 1).unsqueeze(0),
            torch.arange(-(self.imagewidth/2), (self.imagewidth/2), requires_grad=True)[:, None].repeat(1, self.imagewidth).unsqueeze(0)], dim=0).to(device)
    
            
    def forward(self, x):
        x = torch.fft.fftshift(torch.fft.fft2(x))
        weights = torch.complex(self.mlp_real(self.mask), self.mlp_imag(self.mask))
        x = weights.to(self.device)*x
        return x

    
class FreqConv_DW_fftifft(nn.Module):
    '''
    Depthwise convolution inlcuding the transformation into the frequeny domain via FFT 
    and back into the spatial domain via IFFT.
    
    Args:
        planes (int): Number of input channels.
        imageheight (int): Feature map height.
        imagewidth (int): Feature map width.
    '''
    
    def __init__(self, planes, imageheight, imagewidth, act='relu', device='cuda'): #, weights=None):
        super(FreqConv_DW_fftifft, self).__init__()
        self.imageheight  = imageheight
        self.imagewidth = imagewidth
        self.planes = planes
        self.device = device
        self.mlp_imag = MLP(self.planes, act)
        self.mlp_real = MLP(self.planes, act)      
        self.mask = torch.cat([
            torch.arange(-(self.imageheight/2), (self.imageheight/2), requires_grad=True)[None, :].repeat(self.imagewidth, 1).unsqueeze(0),
            torch.arange(-(self.imagewidth/2), (self.imagewidth/2), requires_grad=True)[:, None].repeat(1, self.imagewidth).unsqueeze(0)], dim=0).to(device)
     
    def forward(self, x):
        x = torch.fft.fftshift(torch.fft.fft2(x))
        weights = torch.complex(self.mlp_real(self.mask), self.mlp_imag(self.mask))
        x = weights.to(self.device)*x
        return torch.fft.ifft2(torch.fft.ifftshift(x)).real
    

    
class FreqConv_1x1_ifft(nn.Module):
    '''
    1x1 Convolution inlcuding the transformation into the spatial domain via IFFT.
    
    Args:
        planes (int): Number of input channels.
        out_planes (int): Number of output channels.
    '''
    
    def __init__(self, in_planes, out_planes):
        super(FreqConv_1x1_ifft, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mlp = torch.nn.Linear(in_planes, out_planes, bias=False)
              
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = torch.complex(self.mlp(x.real), self.mlp(x.imag))
        x = x.permute(0, 3, 1, 2)
        return torch.fft.ifft2(torch.fft.ifftshift(x)).real
    
    
class FreqConv_1x1_fftifft(nn.Module):
    '''
    1x1 Convolution inlcuding the transformation into the frequeny domain via FFT 
    and back into the spatial domain via IFFT.
    
    Args:
        planes (int): Number of input channels.
        out_planes (int): Number of output channels.
    '''
    
    def __init__(self, in_planes, out_planes): 
        super(FreqConv_1x1_fftifft, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mlp = torch.nn.Linear(in_planes, out_planes, bias=False)
            
    def forward(self, x):
        x = torch.fft.fft2(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.complex(self.mlp(x.real), self.mlp(x.imag))
        x = x.permute(0, 3, 1, 2)
        return torch.fft.ifft2(x).real

    
    
