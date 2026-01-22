from typing import Optional
import types, gc, os, time, re, platform
import torch
from torch.nn import functional as F
@MyFunction
def cuda_att_one_fp16(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
    kx = torch.empty_like(x)
    vx = torch.empty_like(x)
    rx = torch.empty_like(x)
    k_t = torch.empty((kw.shape[0],), dtype=torch.float32, device=x.device)
    v_t = torch.empty((vw.shape[0],), dtype=torch.float32, device=x.device)
    r_t = torch.empty((rw.shape[0],), dtype=torch.float16, device=x.device)
    x_plus_out_t = torch.empty_like(x)
    t1_t = torch.empty_like(x, dtype=torch.float32)
    t2_t = torch.empty_like(x, dtype=torch.float32)
    p_t = torch.empty_like(x, dtype=torch.float32)
    xx = torch.ops.rwkv.att_one(x, ln_w, ln_b, sx, k_mix, v_mix, r_mix, kw, kx, vw, vx, rw, rx, ow, t_first, k_t, pp, ow, aa, bb, t_decay, v_t, r_t, x_plus_out_t, t1_t, t2_t, p_t)
    return (x_plus_out_t, xx, t1_t, t2_t, p_t)