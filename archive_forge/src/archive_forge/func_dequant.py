import math
import einops
import pytest
import torch
from torch import nn
import bitsandbytes as bnb
from tests.helpers import id_formatter
def dequant(xq, S1, S2, dtype, quant_type):
    if quant_type == 'linear':
        norm = S1 * S2 / (127 * 127)
        return (xq.float() * norm).to(dtype)
    elif quant_type == 'vector':
        x = xq.float()
        if len(xq.shape) == 2 and len(S1.shape) == 3:
            S1 = S1.squeeze(0)
        if len(xq.shape) == 2 and len(S2.shape) == 3:
            S2 = S2.squeeze(0)
        if len(S1.shape) == 2:
            x *= S1.t() / 127
        else:
            x *= S1 / 127
        x *= S2 / 127
        return x.to(dtype)
    else:
        return None