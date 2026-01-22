from typing import Optional
import types, gc, os, time, re, platform
import torch
from torch.nn import functional as F
@MyFunction
def cuda_att_seq_i8(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
    T, C = x.size()
    xx = F.layer_norm(x, (C,), weight=ln_w, bias=ln_b)
    sx = torch.cat((sx.unsqueeze(0), xx[:-1, :]))
    kx = xx * k_mix + sx * (1 - k_mix)
    vx = xx * v_mix + sx * (1 - v_mix)
    rx = xx * r_mix + sx * (1 - r_mix)
    r = torch.sigmoid(self.mm8_seq(rx, rw, rmx, rrx, rmy, rry))
    k = self.mm8_seq(kx, kw, kmx, krx, kmy, kry)
    v = self.mm8_seq(vx, vw, vmx, vrx, vmy, vry)
    y, aa, bb, pp = cuda_wkv(T, C, t_decay, t_first, k, v, aa, bb, pp)
    out = self.mm8_seq(r * y, ow, omx, orx, omy, ory)
    return (x + out, xx[-1, :], aa, bb, pp)