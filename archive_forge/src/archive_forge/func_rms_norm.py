import torch
from torch.nn import init
from flash_attn.ops.layer_norm import (
def rms_norm(x, weight, epsilon):
    return DropoutAddLayerNormFn.apply(x, None, weight, None, None, None, 0.0, epsilon, False, False, True)