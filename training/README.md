# MRI Super-Resolution Training

This directory contains the training script for MRI super-resolution using the UNet3D model.

## Features

The `train_PyTorch.py` script includes the following features similar to the reference super-resolution training code:

- **Comprehensive Logging**: Structured logging with timestamps and different log levels
- **TensorBoard Integration**: Real-time visualization of training metrics and images
- **WandB Support**: Optional Weights & Biases integration for experiment tracking
- **Validation Metrics**: PSNR and SSIM calculation during training
- **Checkpoint Management**: Automatic model saving and resuming from checkpoints
- **Configuration Management**: JSON-based configuration with command-line overrides
- **Image Visualization**: Automatic saving and logging of LR, HR, and SR images

## Usage

### Basic Training

```bash
# Train with default configuration
python train_PyTorch.py

# Train with custom configuration file
python train_PyTorch.py -c config/mri_sr_config.json

# Train with specific GPU
python train_PyTorch.py -gpu 0

# Train with multiple GPUs
python train_PyTorch.py -gpu 0,1
```

### Validation Only

```bash
# Run validation only
python train_PyTorch.py -p val

# Run validation with WandB logging
python train_PyTorch.py -p val -enable_wandb -log_eval
```

### Advanced Options

```bash
# Enable WandB logging
python train_PyTorch.py -enable_wandb

# Enable WandB with checkpoint logging
python train_PyTorch.py -enable_wandb -log_wandb_ckpt

# Debug mode
python train_PyTorch.py -debug
```

## Configuration

The training script uses a JSON configuration file. A sample configuration is provided in `config/mri_sr_config.json`.

### Key Configuration Parameters

- **datasets**: Paths to training and validation data
- **train**: Training parameters (learning rate, iterations, frequencies)
- **path**: Output paths for logs, results, and checkpoints
- **model**: Model configuration
- **wandb**: WandB project settings

### Default Data Structure

```
data/
├── train/
│   ├── low_res/     # Low-resolution .nii files
│   └── high_res/    # High-resolution .nii files
└── val/
    ├── low_res/     # Validation low-res .nii files
    └── high_res/    # Validation high-res .nii files
```

## Output Structure

```
logs/                # Training logs
├── tb_logger/      # TensorBoard logs
└── train_*.log     # Text logs

results/            # Validation images
└── epoch_*/        # Images per epoch
    ├── *_hr.png    # High-resolution images
    ├── *_lr.png    # Low-resolution images
    ├── *_sr.png    # Super-resolution outputs
    └── *_inf.png   # Inference outputs

checkpoints/        # Model checkpoints
├── best_model.pth
└── checkpoint_epoch_*_step_*.pth
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/tb_logger
```

### WandB

If enabled, metrics and images will be automatically logged to your WandB project.

## Model Architecture

The script uses the UNet3D model from `models/Unet_PyTorch.py` with the following features:

- 3D convolutional layers for volumetric data
- Attention mechanisms for improved feature learning
- Residual connections for better gradient flow
- Group normalization for stable training

## Metrics

The following metrics are calculated and logged:

- **Loss**: Mean Squared Error (MSE) between SR and HR images
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index

## Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch
- TensorBoard/TensorBoardX
- scikit-image (for PSNR/SSIM)
- nibabel (for .nii file handling)
- wandb (optional, for experiment tracking) 