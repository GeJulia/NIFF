'''NIFF implementation for small, leight-weight models'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_tiny(nn.Module):
    '''
    NIFF MLP receiving two input feature maps and outputing planes output multiplication weights.
    
    Args:
        planes (int): Desired number of output channels/multiplication weights.
    ''' 
    
    def __init__(self, planes): #, weights=None):
        super(MLP_tiny, self).__init__()
        self.layer_mpl1 = nn.Conv2d(2, 8, 1, padding=0, groups=1)
        self.layer_mpl2 = nn.Conv2d(8, planes, 1, padding=0, groups=1)
        

            
    def forward(self, x):
        x = self.layer_mpl1(x.unsqueeze(0))
        x = F.silu(x)
        x = self.layer_mpl2(x)
        return x


class FreqConv_full_fftifft(nn.Module):
    '''
    Full convolution inlcuding the transformation into the frequeny domain via FFT 
    and back into the spatial domain via IFFT.
    
    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
    '''
    
    def __init__(self, in_planes, out_planes, device='cuda'):
        super(FreqConv_full_fftifft, self).__init__()
        self.device = device
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mlp_imag = MLP_tiny(in_planes*out_planes)
        self.mlp_real = MLP_tiny(in_planes*out_planes)
        self.mask = None
    
    def forward(self, x):
        if self.mask == None:
            self.mask = torch.cat([
            torch.arange(-(x.size(2)/2), (x.size(2)/2), requires_grad=True)[None, :].repeat(x.size(3), 1).unsqueeze(0),
            torch.arange(-(x.size(3)/2), (x.size(3)/2), requires_grad=True)[:, None].repeat(1, x.size(2)).unsqueeze(0)], dim=0).to(self.device)
        x = torch.fft.fftshift(torch.fft.fft2(x))
        weights = torch.complex(self.mlp_real(self.mask), self.mlp_imag(self.mask)).reshape(self.in_planes,self.out_planes,x.size(2),x.size(3))
        x = torch.einsum('bihw,iohw->bohw', [x, weights])
        return torch.fft.ifft2(torch.fft.ifftshift(x)).real
    
    
class FreqConv_full_fft(nn.Module):
    '''
    Full convolution inlcuding the transformation into the frequeny domain via FFT.
    
    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
    '''
    
    def __init__(self, in_planes, out_planes, device='cuda'):
        super(FreqConv_full_fft, self).__init__()
        self.device = device
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mlp_imag = MLP_tiny(in_planes*out_planes)
        self.mlp_real = MLP_tiny(in_planes*out_planes)
        self.mask = None
    
    def forward(self, x):
        if self.mask == None:
            self.mask = torch.cat([
            torch.arange(-(x.size(2)/2), (x.size(2)/2), requires_grad=True)[None, :].repeat(x.size(3), 1).unsqueeze(0),
            torch.arange(-(x.size(3)/2), (x.size(3)/2), requires_grad=True)[:, None].repeat(1, x.size(2)).unsqueeze(0)], dim=0).to(self.device)
        x = torch.fft.fftshift(torch.fft.fft2(x))
        weights = torch.complex(self.mlp_real(self.mask), self.mlp_imag(self.mask)).reshape(self.in_planes,self.out_planes,x.size(2),x.size(3))
        x = torch.einsum('bihw,iohw->bohw', [x, weights])
        return x
    
    
class FreqConv_full_ifft(nn.Module):
    '''
    Full convolution inlcuding the transformation into the spatial domain via IFFT.
    
    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
    '''
    
    def __init__(self, in_planes, out_planes, device='cuda'):
        super(FreqConv_full_ifft, self).__init__()
        self.device = device
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mlp_imag = MLP_tiny(in_planes*out_planes)
        self.mlp_real = MLP_tiny(in_planes*out_planes)
        self.mask = None
    
    def forward(self, x):
        if self.mask == None:
            self.mask = torch.cat([
            torch.arange(-(x.size(2)/2), (x.size(2)/2), requires_grad=True)[None, :].repeat(x.size(3), 1).unsqueeze(0),
            torch.arange(-(x.size(3)/2), (x.size(3)/2), requires_grad=True)[:, None].repeat(1, x.size(2)).unsqueeze(0)], dim=0).to(self.device)
        weights = torch.complex(self.mlp_real(self.mask), self.mlp_imag(self.mask)).reshape(self.in_planes,self.out_planes,x.size(2),x.size(3))
        x = torch.einsum('bihw,iohw->bohw', [x, weights])
        return torch.fft.ifft2(torch.fft.ifftshift(x)).real

    
