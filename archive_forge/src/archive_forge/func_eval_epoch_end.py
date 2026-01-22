from enum import Enum
from typing import Dict, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
from xformers.components import build_attention
from xformers.components.multi_head_dispatch import MultiHeadDispatchConfig
from xformers.factory import xFormer, xFormerConfig, xFormerEncoderConfig
from xformers.utils import generate_matching_config
def eval_epoch_end(self, outputs, prefix: str='train'):
    logs = {}
    counts = torch.tensor([x['count'] for x in outputs]).float()
    logs['count'] = counts.sum()
    for k in ('accu', 'loss'):
        logs[k] = (torch.tensor([x[k] for x in outputs]) * counts).sum() / logs['count']
        self.log(f'{prefix}_{k}_mean', logs[k], sync_dist=True)
    return logs