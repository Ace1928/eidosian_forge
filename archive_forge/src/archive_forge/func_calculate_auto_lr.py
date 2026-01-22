import argparse
import os
import math
import json
from functools import partial
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import tqdm
import wandb
import numpy as np
from ochat.config import MODEL_CONFIG_MAP
from ochat.training_deepspeed.openchat_dataset import OpenchatDataset
def calculate_auto_lr(lr, batch_max_len, model_type, train_dataset):
    if lr is not None:
        return lr
    base_lr = 0.0003
    base_bs = 4000000
    if 'mistral' in model_type.lower():
        base_lr /= 6.0
    elif 'gemma' in model_type.lower():
        base_lr /= 5.5
    loss_weights = np.concatenate(train_dataset.dataset['nz_shifted_loss_weights'])
    supervised_ratio = np.sum(loss_weights != 0) / len(loss_weights)
    supervised_tokens = batch_max_len * dist.get_world_size() * supervised_ratio
    lr = base_lr * math.sqrt(supervised_tokens / base_bs)
    print(f'Use automatic learning rate {lr} (estimated from supervised ratio {supervised_ratio} effective batch size {supervised_tokens})')
    return lr