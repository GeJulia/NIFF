'''NIFF implementation for large models'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''
    NIFF MLP receiving two input feature maps and outputing planes output feature maps
    
    Args:
        planes (int): Desired number of output channels/multiplication weights.
    ''' 
    
    def __init__(self, planes): 
        super(MLP, self).__init__()
        self.layer_mpl1 = nn.Conv2d(2, 16, 1, padding=0, groups=1)
        self.layer_mpl2 = nn.Conv2d(16, 128, 1, padding=0, groups=1)
        self.layer_mpl3 = nn.Conv2d(128, 32, 1, padding=0, groups=1)
        self.layer_mpl4 = nn.Conv2d(32, planes, 1, padding=0, groups=1)
        
    def forward(self, x):
        x = self.layer_mpl1(x.unsqueeze(0))
        x = F.silu(x)
        x = self.layer_mpl2(x)
        x = F.silu(x)
        x = self.layer_mpl3(x)
        x = F.silu(x)
        x = self.layer_mpl4(x)
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
    
    def __init__(self, planes, imageheight, imagewidth): 
        super(FreqConv_DW_fftifft, self).__init__()
        self.imageheight  = imageheight
        self.imagewidth = imagewidth
        self.planes = planes
        self.mlp_imag = MLP(self.planes)
        self.mlp_real = MLP(self.planes)
        self.mask = torch.cat([
            torch.arange(-(self.imageheight/2), (self.imageheight/2), requires_grad=True)[None, :].repeat(self.imagewidth, 1).unsqueeze(0),
            torch.arange(-(self.imagewidth/2), (self.imagewidth/2), requires_grad=True)[:, None].repeat(1, self.imagewidth).unsqueeze(0)], dim=0).cuda()
      
    def forward(self, x):
        x = torch.fft.fftshift(torch.fft.fft2(x))
        weights = torch.complex(self.mlp_real(self.mask), self.mlp_imag(self.mask))
        x = weights.cuda()*x
        return torch.fft.ifft2(torch.fft.ifftshift(x)).real 
    
    
class FreqConv_DW_fft(nn.Module):
    '''
    Depthwise convolution inlcuding only the tranformation into the frequency domain via FFT.
    
    Args:
        planes (int): Number of input channels.
        imageheight (int): Feature map height.
        imagewidth (int): Feature map width.
    '''
    
    def __init__(self, planes, imageheight, imagewidth, device='cuda'): #, weights=None):
        super(FreqConv_DW_fft, self).__init__()
        self.imageheight  = imageheight
        self.imagewidth = imagewidth
        self.planes = planes
        self.device = device
        self.mlp_imag = MLP(self.planes)
        self.mlp_real = MLP(self.planes)
        self.mask = torch.cat([
            torch.arange(-(self.imageheight/2), (self.imageheight/2), requires_grad=True)[None, :].repeat(self.imagewidth, 1).unsqueeze(0),
            torch.arange(-(self.imagewidth/2), (self.imagewidth/2), requires_grad=True)[:, None].repeat(1, self.imagewidth).unsqueeze(0)], dim=0).to(device)
    
    def forward(self, x):
        x = torch.fft.fftshift(torch.fft.fft2(x))
        weights = torch.complex(self.mlp_real(self.mask), self.mlp_imag(self.mask))
        x = weights.to(self.device)*x
        return x


class FreqConv_1x1_fftifft_convnnext(nn.Module):
    '''
    1x1 Convolution inlcuding the transformation into the frequeny domain via FFT 
    and back into the spatial domain via IFFT.
    Inclduing additional tensor permutations needed for ConvNeXt.
    
    Args:
        planes (int): Number of input channels.
        out_planes (int): Number of output channels.
    '''
    
    def __init__(self, in_planes, out_planes): 
        super(FreqConv_1x1_fftifft_convnnext, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mlp = torch.nn.Linear(in_planes, out_planes, bias=False) 
                  
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = torch.fft.fft2(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.complex(self.mlp(x.real), self.mlp(x.imag))
        x = x.permute(0, 3, 1, 2)
        x = torch.fft.ifft2(x).real
        x = x.permute(0, 2, 3, 1)
        return x
    
    
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
        x_real = self.mlp(x.real)
        x_imag = self.mlp(x.imag)
        x = torch.complex(x_real, x_imag)
        x = x.permute(0, 3, 1, 2)
        return torch.fft.ifft2(x).real
    
    
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
        x_real = self.mlp(x.real)
        x_imag = self.mlp(x.imag)
        x = torch.complex(x_real, x_imag)
        x = x.permute(0, 3, 1, 2)
        return torch.fft.ifft2(torch.fft.ifftshift(x)).real