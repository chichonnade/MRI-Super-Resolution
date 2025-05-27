import logging
import json
import os
from collections import OrderedDict


class NoneDict(dict):
    """Dictionary that returns None for missing keys instead of raising KeyError."""
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    """Convert a dictionary to NoneDict recursively."""
    if isinstance(opt, dict):
        new_opt = NoneDict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return new_opt
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def parse(args):
    """Parse configuration from JSON file."""
    if hasattr(args, 'config') and args.config:
        with open(args.config, 'r') as f:
            opt = json.load(f, object_pairs_hook=OrderedDict)
    else:
        # Default configuration for MRI super-resolution
        opt = {
            'path': {
                'log': 'logs',
                'tb_logger': 'logs/tb_logger',
                'results': 'results',
                'resume_state': None
            },
            'datasets': {
                'train': {
                    'lr_dir': 'data/train/low_res',
                    'hr_dir': 'data/train/high_res',
                    'batch_size': 4,
                    'num_workers': 4
                },
                'val': {
                    'lr_dir': 'data/val/low_res', 
                    'hr_dir': 'data/val/high_res',
                    'batch_size': 1,
                    'num_workers': 1
                }
            },
            'train': {
                'n_iter': 100000,
                'print_freq': 100,
                'val_freq': 1000,
                'save_checkpoint_freq': 5000,
                'lr': 1e-4
            },
            'model': {
                'name': 'UNet3D'
            },
            'phase': 'train',
            'enable_wandb': False,
            'log_wandb_ckpt': False,
            'log_eval': False
        }
    
    # Update with command line arguments
    if hasattr(args, 'phase'):
        opt['phase'] = args.phase
    if hasattr(args, 'enable_wandb'):
        opt['enable_wandb'] = args.enable_wandb
    if hasattr(args, 'log_wandb_ckpt'):
        opt['log_wandb_ckpt'] = args.log_wandb_ckpt
    if hasattr(args, 'log_eval'):
        opt['log_eval'] = args.log_eval
    
    return opt


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    """Set up logger."""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def dict2str(opt, indent_l=1):
    """Convert dictionary to string for logging."""
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def get_timestamp():
    """Get timestamp string."""
    import time
    return time.strftime('%y%m%d_%H%M%S', time.localtime()) 