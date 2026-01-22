import itertools
import random
from functools import partial
import torch
from torch.utils import benchmark
import xformers.ops
import xformers.ops.fmha as fmha
from xformers.attn_bias_utils import create_attn_bias
from xformers.benchmarks.utils import benchmark_main_helper
def ref_attention_mqa(q, k, v, attn_bias=None, drop_mask=None, p=0.0, scale=None, dtype=None):
    if q.ndim == 4:
        B, M, Hq, K = q.shape
        _, N, Hkv, Kv = v.shape
        nhead_ratio_qk = Hq // Hkv

        def attn_bias_head(head: int):
            if isinstance(attn_bias, torch.Tensor):
                assert attn_bias.ndim == 4
                _, H, _, _ = attn_bias.shape
                assert H == Hq
                bias_bghmn = attn_bias.reshape(B, Hkv, nhead_ratio_qk, M, N)
                return bias_bghmn[:, :, head]
            if isinstance(attn_bias, fmha.attn_bias.LowerTriangularMaskWithTensorBias):
                assert attn_bias._bias.ndim == 4
                _, H, _, _ = attn_bias._bias.shape
                assert H == Hq
                bias_bghmn = attn_bias._bias.reshape(B, Hkv, nhead_ratio_qk, M, N)
                return fmha.attn_bias.LowerTriangularMaskWithTensorBias(bias_bghmn[:, :, head])
            return attn_bias
        q_bmghk = q.reshape((B, M, Hkv, nhead_ratio_qk, K))
        return torch.stack([ref_attention_bmhk(q_bmghk[:, :, :, h], k, v, attn_bias=attn_bias_head(h), dtype=dtype) for h in range(q_bmghk.shape[3])], dim=3).reshape((B, M, Hq, Kv))
    assert q.ndim == 3
    if dtype is None:
        dtype = torch.float32
    q = q.to(dtype=dtype)
    k = k.to(dtype=dtype)
    v = v.to(dtype=dtype)
    scale = scale if scale is not None else q.shape[-1] ** (-0.5)
    q = q * scale
    attn = q @ k.transpose(-2, -1)
    if attn_bias is not None:
        if isinstance(attn_bias, xformers.ops.AttentionBias):
            attn_bias_tensor = attn_bias.materialize((q.shape[0], 1, q.shape[1], k.shape[1]), device=q.device, dtype=dtype)
        else:
            attn_bias_tensor = attn_bias.to(dtype=dtype)
        if attn_bias_tensor.ndim == 4:
            assert q.shape[0] == attn_bias_tensor.shape[0] * attn_bias_tensor.shape[1]
            attn_bias_tensor = attn_bias_tensor.reshape([-1, *attn_bias_tensor.shape[2:]])
        attn = attn + attn_bias_tensor
    attn = attn.softmax(-1)
    if drop_mask is not None:
        attn = attn * (drop_mask / (1 - p))
    return attn @ v