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
def setup_log(args, attention_name, task) -> Tuple[str, TensorBoardLogger]:
    experiment_name = f'{task}__{attention_name}'
    logger = TensorBoardLogger(save_dir=args.checkpoint_dir, name='', version=experiment_name)
    log_dir = os.path.join(logger._save_dir, experiment_name)
    return (log_dir, logger)