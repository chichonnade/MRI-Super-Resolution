import numpy as np
import torch
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import nibabel as nib
import os


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Convert a torch tensor to numpy image array.
    
    Args:
        tensor: Input tensor (C, H, W) or (H, W) or (B, C, H, W)
        out_type: Output data type
        min_max: Min and max values for normalization
        
    Returns:
        numpy array: Converted image
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    
    if tensor.dim() == 3:
        # For 3D tensors, take the middle slice
        if tensor.shape[0] > 1:  # Multiple channels
            tensor = tensor[0]  # Take first channel
        else:
            tensor = tensor.squeeze(0)
        
        # Take middle slice if it's a 3D volume
        if tensor.dim() == 3:
            mid_slice = tensor.shape[0] // 2
            tensor = tensor[mid_slice]
    
    elif tensor.dim() == 4:
        # For 4D tensors (batch), take first item and middle slice
        tensor = tensor[0]
        if tensor.shape[0] > 1:
            tensor = tensor[0]
        else:
            tensor = tensor.squeeze(0)
        
        if tensor.dim() == 3:
            mid_slice = tensor.shape[0] // 2
            tensor = tensor[mid_slice]
    
    # Convert to numpy and scale to output type range
    img_np = tensor.numpy()
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round().astype(np.uint8)
    
    return img_np


def save_img(img, img_path):
    """Save image to file."""
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    
    if img_path.endswith('.nii') or img_path.endswith('.nii.gz'):
        # Save as NIfTI file
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)  # Add depth dimension
        nii_img = nib.Nifti1Image(img, affine=np.eye(4))
        nib.save(nii_img, img_path)
    else:
        # Save as regular image
        cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2, max_value=255):
    """Calculate PSNR between two images."""
    if isinstance(img1, torch.Tensor):
        img1 = tensor2img(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2img(img2)
    
    return psnr(img1, img2, data_range=max_value)


def calculate_ssim(img1, img2, max_value=255):
    """Calculate SSIM between two images."""
    if isinstance(img1, torch.Tensor):
        img1 = tensor2img(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2img(img2)
    
    return ssim(img1, img2, data_range=max_value)


def calculate_mse(img1, img2):
    """Calculate MSE between two images."""
    if isinstance(img1, torch.Tensor):
        img1 = tensor2img(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2img(img2)
    
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2) 