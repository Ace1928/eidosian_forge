import os
from typing import Any, Dict
import torch
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components.attention.attention_mask import AttentionMask
from xformers.components.attention.core import scaled_dot_product_attention
def sdp_attention():
    with torch.cuda.amp.autocast(enabled=use_amp):
        y = scaled_dot_product_attention(q=q, k=k, v=v, att_mask=m_custom, block_size=BS)
        if backward:
            torch.norm(y).backward()
        return y