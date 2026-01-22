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
def create_lr_scheduler(args, train_total_steps):
    lr_scheduler = partial(cosine_schedule_with_warmup_lr_lambda, num_warmup_steps=round(args.lr_warmup_ratio * train_total_steps), num_training_steps=train_total_steps, min_ratio=args.lr_min_ratio)
    return lr_scheduler