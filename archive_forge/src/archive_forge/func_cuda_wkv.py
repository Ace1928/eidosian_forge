import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
from torch import nn
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
@MyStatic
def cuda_wkv(T: int, C: int, w, u, k, v, aa, bb, pp):
    assert 1 * C % min(C, 32) == 0
    assert k.dtype == v.dtype == torch.float16 or k.dtype == v.dtype == torch.float32
    assert w.dtype == u.dtype == aa.dtype == bb.dtype == pp.dtype == torch.float32
    w = w.contiguous()
    u = u.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    y = torch.empty((T, C), device=w.device, memory_format=torch.contiguous_format, dtype=k.dtype)
    torch.ops.rwkv.wkv_forward(1, T, C, w, u, k, v, y, aa, bb, pp)
    return (y, aa, bb, pp)