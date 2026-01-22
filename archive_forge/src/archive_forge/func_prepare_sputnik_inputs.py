import gc
import math
from collections import namedtuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
import triton
from triton.ops.blocksparse import matmul as blocksparse_matmul
from xformers.benchmarks.utils import pretty_barplot
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import SparseCS, _matmul_with_mask
def prepare_sputnik_inputs(self, query, key, config, mask):
    mask_cs = torch.ones([config.batch_size, config.num_heads, config.seq_length, config.seq_length], dtype=torch.bool, device='cuda')
    mask_cs = triton.testing.mask_tensor(mask_cs, mask, config.block_size, value=False)
    query_cs = query.flatten(start_dim=0, end_dim=1).to(torch.float32)
    key_cs = key.flatten(start_dim=0, end_dim=1).to(torch.float32)
    query_cs = query_cs.contiguous()
    key_cs = key_cs.transpose(-2, -1)
    sparse_mask_cs = SparseCS(mask_cs.flatten(start_dim=0, end_dim=1).contiguous(), device=torch.device('cuda'))
    return (query_cs, key_cs, sparse_mask_cs)