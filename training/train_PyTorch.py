import torch
import sys
import os
import argparse
import logging
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import data.data as Data
import models.model as Model
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # Parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which returns None for missing key
    opt = Logger.dict_to_nonedict(opt)

    # Logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Create directories
    os.makedirs(opt['path']['log'], exist_ok=True)
    os.makedirs(opt['path']['tb_logger'], exist_ok=True)
    os.makedirs(opt['path']['results'], exist_ok=True)

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # Determine device
    if args.gpu_ids and torch.cuda.is_available():
        device_ids = [int(id) for id in args.gpu_ids.split(',')]
        device = torch.device(f'cuda:{device_ids[0]}')
        torch.cuda.set_device(device)
        device_message = f"Using CUDA on GPU(s): {device_ids}"
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        device_message = "Using default CUDA device."
    else:
        device = torch.device('cpu')
        device_message = "Using CPU for computation."

    # Log the device information
    logger.info(f"Device selected for training: {device_message}")
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    logger.info("Starting dataset initialization...")
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info("Initial Dataset Finished")

    # Model
    model = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = model.begin_step
    current_epoch = model.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    if opt['phase'] == 'train':
        logger.info('Begin Training.')
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                
                model.feed_data(train_data)
                model.optimize_parameters()
                
                # Log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = model.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # Validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        model.feed_data(val_data)
                        model.test(continous=False)
                        visuals = model.get_current_visuals()
                        
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
                        
                        # Save images
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        
                        # Ensure all images have the same format
                        def preprocess_image(img):
                            if img.ndim == 2:  # Grayscale image (H, W)
                                return np.expand_dims(img, axis=-1)  # Add channel dimension
                            elif img.ndim == 3:  # RGB image (H, W, 3)
                                return img
                            else:
                                raise ValueError(f"Unsupported image dimensions: {img.shape}")

                        # Preprocess images
                        fake_img_proc = preprocess_image(fake_img)
                        sr_img_proc = preprocess_image(sr_img)
                        hr_img_proc = preprocess_image(hr_img)

                        # Concatenate for tensorboard
                        try:
                            concatenated_images = np.concatenate((fake_img_proc, sr_img_proc, hr_img_proc), axis=1)
                            if concatenated_images.ndim == 3 and concatenated_images.shape[-1] <= 3:
                                tb_logger.add_image(
                                    'Iter_{}'.format(current_step),
                                    np.transpose(concatenated_images, [2, 0, 1]),
                                    idx)
                        except Exception as e:
                            logger.warning(f"Could not log image to tensorboard: {e}")

                        # Calculate metrics
                        psnr_val = Metrics.calculate_psnr(sr_img, hr_img)
                        ssim_val = Metrics.calculate_ssim(sr_img, hr_img)
                        
                        avg_psnr += psnr_val
                        avg_ssim += ssim_val

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((fake_img_proc, sr_img_proc, hr_img_proc), axis=1)
                            )

                    avg_psnr = avg_psnr / idx if idx > 0 else 0
                    avg_ssim = avg_ssim / idx if idx > 0 else 0
                    
                    # Log validation results
                    logger.info('# Validation # PSNR: {:.4e}, SSIM: {:.4e}'.format(avg_psnr, avg_ssim))
                    logger_val = logging.getLogger('val')
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr, avg_ssim))
                    
                    # Tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_ssim': avg_ssim,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                # Save checkpoint
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    model.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # Save final model
        logger.info('End of training.')
        model.save_network(current_epoch, current_step)
        
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        
        for _, val_data in enumerate(val_loader):
            idx += 1
            model.feed_data(val_data)
            model.test(continous=True)
            visuals = model.get_current_visuals()

            # Convert visuals to images
            hr_img = Metrics.tensor2img(visuals['HR'])
            lr_img = Metrics.tensor2img(visuals['LR'])
            fake_img = Metrics.tensor2img(visuals['INF'])
            sr_img = Metrics.tensor2img(visuals['SR'])

            # Save images
            Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
            Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            # Calculate metrics
            eval_psnr = Metrics.calculate_psnr(sr_img, hr_img)
            eval_ssim = Metrics.calculate_ssim(sr_img, hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            # Log evaluation data to WandB if enabled
            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, sr_img, hr_img, eval_psnr, eval_ssim)
        
        avg_psnr = avg_psnr / idx if idx > 0 else 0
        avg_ssim = avg_ssim / idx if idx > 0 else 0

        # Log final results
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })

    # Cleanup
    if wandb_logger:
        wandb_logger.finish()