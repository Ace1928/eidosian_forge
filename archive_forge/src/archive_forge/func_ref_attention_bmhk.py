import itertools
import random
from functools import partial
import torch
from torch.utils import benchmark
import xformers.ops
import xformers.ops.fmha as fmha
from xformers.attn_bias_utils import create_attn_bias
from xformers.benchmarks.utils import benchmark_main_helper
def ref_attention_bmhk(q, k, v, attn_bias, scale=None, dtype=None) -> torch.Tensor:
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape([t.shape[0] * t.shape[2], t.shape[1], t.shape[3]])
    if isinstance(attn_bias, xformers.ops.AttentionBias):
        attn_bias = attn_bias.materialize((q.shape[0], q.shape[2], q.shape[1], k.shape[1]), device=q.device, dtype=torch.float32).reshape([q.shape[0] * q.shape[2], q.shape[1], k.shape[1]])
    out = ref_attention_mqa(T(q), T(k), T(v), attn_bias, scale=scale, dtype=dtype)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))