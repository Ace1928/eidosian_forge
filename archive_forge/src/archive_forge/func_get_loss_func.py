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
def get_loss_func(self):

    def loss_func(loss_mask, output_tensor):
        lm_loss_ = output_tensor.float()
        lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group([lm_loss])
        return (loss, {'lm loss': averaged_losses[0]})
    return loss_func