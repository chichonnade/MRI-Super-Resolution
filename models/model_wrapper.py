import torch
import torch.nn as nn
import torch.optim as optim
import os
from .unet_2d import UNetSR


class UNet3DWrapper:
    """Wrapper class for UNet2D model to provide a unified interface for MRI super-resolution."""
    
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize 2D UNet model for slice-by-slice processing
        self.model = UNetSR(
            in_channel=1,
            out_channel=1,
            inner_channel=32,
            norm_groups=8,
            channel_mults=(1, 2, 4, 8),
            attn_res=(16, 32),
            res_blocks=2,
            dropout=0,
            image_size=64,
            scale_factor=2  # 2x super-resolution
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=opt['train']['lr']
        )
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.begin_step = 0
        self.begin_epoch = 0
        
        # Load checkpoint if specified
        if opt['path']['resume_state']:
            self.load_network(opt['path']['resume_state'])
    
    def feed_data(self, data):
        """Feed data to the model."""
        self.lr_img, self.hr_img = data
        self.lr_img = self.lr_img.to(self.device)
        self.hr_img = self.hr_img.to(self.device)
    
    def _process_3d_volume(self, volume):
        """Process 3D volume slice by slice using 2D UNet."""
        b, c, d, h, w = volume.shape
        sr_slices = []
        
        for i in range(d):
            # Extract 2D slice
            slice_2d = volume[:, :, i, :, :]  # (B, C, H, W)
            
            # Process with 2D UNet
            sr_slice = self.model(slice_2d)
            sr_slices.append(sr_slice)
        
        # Stack slices back to 3D volume
        sr_volume = torch.stack(sr_slices, dim=2)  # (B, C, D, H, W)
        return sr_volume
    
    def optimize_parameters(self):
        """Optimize model parameters."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass - process 3D volume slice by slice
        self.sr_img = self._process_3d_volume(self.lr_img)
        
        # Calculate loss
        self.loss = self.criterion(self.sr_img, self.hr_img)
        
        # Backward pass
        self.loss.backward()
        self.optimizer.step()
    
    def test(self, continous=False):
        """Test the model."""
        self.model.eval()
        with torch.no_grad():
            self.sr_img = self._process_3d_volume(self.lr_img)
    
    def get_current_log(self):
        """Get current training logs."""
        return {'loss': self.loss.item()}
    
    def get_current_visuals(self):
        """Get current visual results."""
        return {
            'LR': self.lr_img.detach().cpu(),
            'HR': self.hr_img.detach().cpu(),
            'SR': self.sr_img.detach().cpu(),
            'INF': self.sr_img.detach().cpu()  # For compatibility with reference code
        }
    
    def save_network(self, epoch, step):
        """Save model checkpoint."""
        save_path = os.path.join(self.opt['path']['log'], f'checkpoint_epoch_{epoch}_step_{step}.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'step': step
        }
        
        torch.save(state_dict, save_path)
        
        # Also save as best model if it's the latest
        best_path = os.path.join(self.opt['path']['log'], 'best_model.pth')
        torch.save(state_dict, best_path)
    
    def load_network(self, load_path):
        """Load model checkpoint."""
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.begin_epoch = checkpoint.get('epoch', 0)
            self.begin_step = checkpoint.get('step', 0)
            
            print(f'Loaded checkpoint from {load_path}')
        else:
            print(f'Checkpoint not found at {load_path}')


def create_model(opt):
    """Create model instance."""
    return UNet3DWrapper(opt) 