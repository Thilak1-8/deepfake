import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import pywt
from scipy.fftpack import dct

class AdvancedTransform:
    """Custom transform that applies advanced feature extraction techniques"""
    
    def __init__(self, img_size=224):
        self.img_size = img_size
        
    def __call__(self, img, return_features=False):
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Resize
        img_resized = cv2.resize(img_np, (self.img_size, self.img_size))
        
        # 1. Color Space Conversion to YCbCr
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_ycbcr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2YCrCb)
        else:
            img_ycbcr = img_resized
        
        # 2. FFT Magnitude (on Y channel)
        y_channel = img_ycbcr[:, :, 0] if len(img_ycbcr.shape) == 3 else img_ycbcr
        fft = np.fft.fft2(y_channel)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        magnitude_spectrum = np.log(magnitude_spectrum + 1)  # Log scale
        magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
        
        # 3. DCT Coefficients (on Y channel)
        dct_coeff = dct(dct(y_channel.T, norm='ortho').T, norm='ortho')
        dct_coeff = (dct_coeff - dct_coeff.min()) / (dct_coeff.max() - dct_coeff.min())
        
        # 4. Wavelet Transform (on Y channel)
        coeffs = pywt.dwt2(y_channel, 'haar')
        cA, (cH, cV, cD) = coeffs
        wavelet_features = np.abs(cH)  # Using horizontal details
        wavelet_features = cv2.resize(wavelet_features, (self.img_size, self.img_size))
        wavelet_features = (wavelet_features - wavelet_features.min()) / (wavelet_features.max() - wavelet_features.min())
        
        # Stack all features as channels
        # Original YCbCr (3 channels) + FFT (1) + DCT (1) + Wavelet (1) = 6 channels
        fft_resized = cv2.resize(magnitude_spectrum, (self.img_size, self.img_size))
        dct_resized = cv2.resize(dct_coeff, (self.img_size, self.img_size))
        
        combined = np.stack([
            img_ycbcr[:, :, 0] / 255.0,  # Y channel
            img_ycbcr[:, :, 1] / 255.0,  # Cb channel
            img_ycbcr[:, :, 2] / 255.0,  # Cr channel
            fft_resized,
            dct_resized,
            wavelet_features
        ], axis=0)
        
        tensor_output = torch.tensor(combined, dtype=torch.float32)

        if return_features:
            return tensor_output, {
                'original': img_resized,
                'y_channel': img_ycbcr[:, :, 0],
                'fft': fft_resized,
                'dct': dct_resized,
                'wavelet': wavelet_features
            }
        return tensor_output

class AdvancedResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(AdvancedResNet, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer to accept 6 channels instead of 3
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            6,  # 6 input channels (YCbCr + FFT + DCT + Wavelet)
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Initialize new conv1 weights
        with torch.no_grad():
            # Copy weights from original 3 channels and duplicate for other 3
            if pretrained:
                self.resnet.conv1.weight[:, :3, :, :] = original_conv1.weight
                self.resnet.conv1.weight[:, 3:, :, :] = original_conv1.weight
        
        # Modify final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)
