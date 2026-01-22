import functools
import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.cpp_extension import load
@MyFunction
def jit_func_2(self, x, g):
    B, T, C = x.size()
    x = x.view(B * T, C)
    x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
    x = self.output(x * g)
    return x