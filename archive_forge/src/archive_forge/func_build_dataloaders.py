import argparse
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, cast
import pytorch_lightning as pl
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from xformers.benchmarks.LRA.code.dataset import LRADataset
from xformers.benchmarks.LRA.code.model_wrapper import ModelForSC, ModelForSCDual
from xformers.components.attention import ATTENTION_REGISTRY
def build_dataloaders(args: argparse.Namespace, config_training: Dict, num_workers: int=4) -> Dict[str, DataLoader]:
    datasets = {}
    for component in ('train', 'dev', 'test'):
        datasets[component] = LRADataset(file_path=f'datasets/{args.task}.{component}.pickle', seq_len=config_training['seq_len'])
    accumu_steps = config_training['gradient_accumulation']
    logging.info(f'accumu_steps={accumu_steps}')
    per_gpu_batch_size = config_training['batch_size'] // args.world_size // accumu_steps
    logging.warning(f'Requested batch size: {config_training['batch_size']}. Given world            size and grad accumulation, per-gpu batch is            {per_gpu_batch_size}')
    dataloaders = {k: DataLoader(v, batch_size=per_gpu_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers) for k, v in datasets.items()}
    return dataloaders