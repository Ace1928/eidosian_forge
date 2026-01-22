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
def loss_func_finetune(labels, logits):
    if num_labels == 1:
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
    elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    else:
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
    averaged_losses = average_losses_across_data_parallel_group([loss])
    return (loss, {'loss': averaged_losses[0]})