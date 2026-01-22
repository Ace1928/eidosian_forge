import math
import torch
import torch.nn as nn
import torch.nn.functional as F
@torch.jit.script
def bias_gelu_back(g, y, bias):
    """Assume that y has shape (B, D) and bias has shape (D)"""
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    grad_y = ff * g
    return (grad_y.to(dtype=y.dtype), grad_y.sum(dim=0, dtype=bias.dtype))