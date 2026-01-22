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
def att_seq_v5_1(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
    xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
    sx = torch.cat((sx.unsqueeze(0), xx[:-1, :]))
    kx = xx * k_mix + sx * (1 - k_mix)
    vx = xx * v_mix + sx * (1 - v_mix)
    rx = xx * r_mix + sx * (1 - r_mix)
    gx = xx * g_mix + sx * (1 - g_mix)
    H = t_decay.shape[0]
    N = x.shape[-1] // H
    T = x.shape[0]
    w = t_decay.reshape(-1, 1)
    u = t_first.reshape(-1, 1)
    ws = w.pow(T).reshape(H, 1, 1)
    ind = torch.arange(T - 1, -1, -1, device=w.device).unsqueeze(0).repeat(H, 1)
    w = w.repeat(1, T).pow(ind)
    wk = w.reshape(H, 1, T)
    wb = wk.transpose(-2, -1).flip(1)
    w = torch.cat([w[:, 1:], u], dim=1)
    w = F.pad(w, (0, T))
    w = torch.tile(w, [T])
    w = w[:, :-T].reshape(-1, T, 2 * T - 1)
    w = w[:, :, T - 1:].reshape(H, T, T)
    r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
    k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
    v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
    g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))
    out = r @ k * w @ v + r @ s * wb
    s = ws * s + k * wk @ v
    out = out.transpose(0, 1).contiguous().reshape(T, H * N)
    out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps=0.00064)
    out = out.to(dtype=x.dtype) * g
    out = matmul(out, ow, omx, orx, omy, ory)
    return (x + out, xx[-1, :], s)