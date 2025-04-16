import os
import random
import numpy as np
import torch
import yaml

def load_config(config_path):
    """Loads YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seeds(seed):
    """Sets random seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Potentially make CuDNN deterministic, but might impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Seeds set to {seed}")

def get_latest_checkpoint(log_dir, model_name="rl_model"):
    """Finds the latest saved model checkpoint."""
    checkpoints_dir = os.path.join(log_dir, model_name)
    if not os.path.exists(checkpoints_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith(model_name) and f.endswith(".zip")]
    if not checkpoints:
        return None
    # Extract step numbers and find the max
    steps = [int(f.split('_')[-2]) for f in checkpoints]
    latest_step = max(steps)
    latest_checkpoint = os.path.join(checkpoints_dir, f"{model_name}_{latest_step}_steps.zip")
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint