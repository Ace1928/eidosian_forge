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
def v5_2_before(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
    xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
    sx = torch.cat((sx.unsqueeze(0), xx[:-1, :]))
    kx = xx * k_mix + sx * (1 - k_mix)
    vx = xx * v_mix + sx * (1 - v_mix)
    rx = xx * r_mix + sx * (1 - r_mix)
    gx = xx * g_mix + sx * (1 - g_mix)
    r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
    k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
    v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
    g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))
    return (r, k, v, g, xx[-1, :], s.transpose(-1, -2).contiguous())