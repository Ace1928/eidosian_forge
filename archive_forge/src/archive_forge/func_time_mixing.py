import numpy as np
import types, torch
from torch.nn import functional as F
from tokenizers import Tokenizer
@torch.jit.script_method
def time_mixing(self, x, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
    xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
    xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
    xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)
    state[5 * i + 1] = x
    r = torch.sigmoid(rw @ xr)
    k = kw @ xk
    v = vw @ xv
    aa = state[5 * i + 2]
    bb = state[5 * i + 3]
    pp = state[5 * i + 4]
    ww = time_first + k
    qq = torch.maximum(pp, ww)
    e1 = torch.exp(pp - qq)
    e2 = torch.exp(ww - qq)
    a = e1 * aa + e2 * v
    b = e1 * bb + e2
    wkv = a / b
    ww = pp + time_decay
    qq = torch.maximum(ww, k)
    e1 = torch.exp(ww - qq)
    e2 = torch.exp(k - qq)
    state[5 * i + 2] = e1 * aa + e2 * v
    state[5 * i + 3] = e1 * bb + e2
    state[5 * i + 4] = qq
    return ow @ (r * wkv)