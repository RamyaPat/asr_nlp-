import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import time


def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_dir: str,
    is_best: bool = False
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path: str, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('train_loss', 0), checkpoint.get('val_loss', 0)


def save_config(config: Dict[str, Any], config_path: str):
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()

    @property
    def elapsed(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0


def calculate_rtf(audio_duration: float, processing_time: float) -> float:
    if audio_duration == 0:
        return 0
    return processing_time / audio_duration


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.2f}s"


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_model_info(model, logger=None):
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)

    info = f"""
    Model Information:
    ==================
    Total Parameters: {total_params:,}
    Trainable Parameters: {trainable_params:,}
    Non-trainable Parameters: {total_params - trainable_params:,}
    Trainable Ratio: {100 * trainable_params / total_params:.2f}%
    """

    if logger:
        logger.info(info)
    else:
        print(info)
