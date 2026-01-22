import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
@staticmethod
def get_decoder_mask(seq_length, device):
    attention_mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device))
    attention_mask = attention_mask < 0.5
    return attention_mask