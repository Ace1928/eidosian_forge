import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def vectorwise_quant(x, dim=1, quant_type='vector'):
    if quant_type == 'linear':
        max1 = torch.abs(x).max().float()
        xq = torch.round(x / max1 * 127).to(torch.int8)
        return (xq, max1)
    elif quant_type in ['vector', 'row']:
        max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        xq = torch.round(x * (C / max1)).to(torch.int8)
        return (xq, max1)
    elif quant_type == 'zeropoint':
        dtype = x.dtype
        x = x.float()
        dyna = x.max() - x.min()
        if dyna == 0:
            dyna = 1
        qx = 255.0 / dyna
        minx = x.min()
        zpx = torch.round(minx * qx)
        x = torch.round(qx * x - zpx) + zpx
        return (x, qx)
    elif quant_type in ['vector-zeropoint', 'row-zeropoint']:
        dtype = x.dtype
        x = x.float()
        dyna = torch.amax(x, dim=dim, keepdim=True) - torch.amin(x, dim=dim, keepdim=True)
        dyna[dyna == 0] = 1
        qx = 255.0 / dyna
        minx = torch.amin(x, dim=dim, keepdim=True)
        zpx = torch.round(minx * qx)
        x = torch.round(qx * x - zpx) + zpx
        return (x, qx)
    elif quant_type == 'truncated-vector':
        with torch.no_grad():
            absx = torch.abs(x)
            max1 = torch.amax(absx, dim=dim, keepdim=True)
            max1 = max1 * 0.7
            idx = absx > max1.expand_as(absx)
            sign = torch.sign(x[idx])
            x[idx] = max1.expand_as(absx)[idx] * sign
            xq = torch.round(x / max1 * C).to(torch.int8)
        return (xq, max1)
    else:
        return None