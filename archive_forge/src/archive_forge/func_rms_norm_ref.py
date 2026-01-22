import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
import triton
import triton.language as tl
def rms_norm_ref(x, weight, bias, residual=None, x1=None, weight1=None, bias1=None, eps=1e-06, dropout_p=0.0, rowscale=None, prenorm=False, dropout_mask=None, dropout_mask1=None, upcast=False):
    dtype = x.dtype
    if upcast:
        x = x.float()
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        residual = residual.float() if residual is not None else residual
        x1 = x1.float() if x1 is not None else None
        weight1 = weight1.float() if weight1 is not None else None
        bias1 = bias1.float() if bias1 is not None else None
    if x1 is not None:
        assert rowscale is None, 'rowscale is not supported with parallel LayerNorm'
    if rowscale is not None:
        x = x * rowscale[..., None]
    if dropout_p > 0.0:
        if dropout_mask is not None:
            x = x.masked_fill(~dropout_mask, 0.0) / (1.0 - dropout_p)
        else:
            x = F.dropout(x, p=dropout_p)
        if x1 is not None:
            if dropout_mask1 is not None:
                x1 = x1.masked_fill(~dropout_mask1, 0.0) / (1.0 - dropout_p)
            else:
                x1 = F.dropout(x1, p=dropout_p)
    if x1 is not None:
        x = x + x1
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight + bias if bias is not None else x * rstd * weight).to(dtype)
    if weight1 is None:
        return out if not prenorm else (out, x)
    else:
        out1 = (x * rstd * weight1 + bias1 if bias1 is not None else x * rstd * weight1).to(dtype)
        return (out, out1) if not prenorm else (out, out1, x)