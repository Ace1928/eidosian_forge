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
@MyFunction
def v5_2_after(self, t_decay, out, s, x, xxx, g, lx_w, lx_b, ow, omx, orx, omy, ory):
    H = t_decay.shape[0]
    N = x.shape[-1] // H
    T = x.shape[0]
    s = s.transpose(-1, -2)
    out = out.reshape(T, H * N)
    out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps=0.00064)
    out = out.to(dtype=x.dtype) * g
    out = matmul(out, ow, omx, orx, omy, ory)
    return (x + out, xxx, s)