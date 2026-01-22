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
def att_one(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
    xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
    kx = xx * k_mix + sx * (1 - k_mix)
    vx = xx * v_mix + sx * (1 - v_mix)
    rx = xx * r_mix + sx * (1 - r_mix)
    r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
    k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
    v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
    ww = t_first + k
    p = torch.maximum(pp, ww)
    e1 = torch.exp(pp - p)
    e2 = torch.exp(ww - p)
    wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=x.dtype)
    ww = t_decay + pp
    p = torch.maximum(ww, k)
    e1 = torch.exp(ww - p)
    e2 = torch.exp(k - p)
    out = matmul(r * wkv, ow, omx, orx, omy, ory)
    return (x + out, xx, e1 * aa + e2 * v, e1 * bb + e2, p)