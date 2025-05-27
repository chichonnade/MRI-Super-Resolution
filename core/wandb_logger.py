import wandb
import numpy as np
import os


class WandbLogger:
    def __init__(self, opt):
        """Initialize WandB logger."""
        self.opt = opt
        
        # Initialize wandb
        wandb.init(
            project=opt.get('wandb_project', 'mri-super-resolution'),
            name=opt.get('wandb_run_name', None),
            config=opt,
            resume=opt.get('wandb_resume', False)
        )
        
        self.eval_table = wandb.Table(columns=['Image', 'PSNR', 'SSIM'])
        
    def log_metrics(self, metrics_dict):
        """Log metrics to WandB."""
        wandb.log(metrics_dict)
    
    def log_image(self, key, image, caption=None):
        """Log image to WandB."""
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] == 1:
                image = image.squeeze(0)  # Remove channel dimension for grayscale
            wandb.log({key: wandb.Image(image, caption=caption)})
    
    def log_checkpoint(self, epoch, step):
        """Log model checkpoint to WandB."""
        checkpoint_path = f'checkpoint_epoch_{epoch}_step_{step}.pth'
        if os.path.exists(checkpoint_path):
            wandb.save(checkpoint_path)
    
    def log_eval_data(self, lr_img, sr_img, hr_img, psnr_val, ssim_val):
        """Log evaluation data to WandB table."""
        # Create a combined image for visualization
        combined_img = np.concatenate([lr_img, sr_img, hr_img], axis=1)
        
        self.eval_table.add_data(
            wandb.Image(combined_img, caption=f"LR | SR | HR"),
            psnr_val,
            ssim_val
        )
    
    def log_eval_table(self):
        """Log the evaluation table to WandB."""
        wandb.log({"evaluation_results": self.eval_table})
    
    def finish(self):
        """Finish WandB run."""
        wandb.finish() 