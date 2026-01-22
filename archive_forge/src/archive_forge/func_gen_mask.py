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
def gen_mask(self, keep_blocked=True):
    seq_length = self.config.seq_length
    if keep_blocked:
        seq_length = self.config.blocked_seq_length
    H = int(math.sqrt(seq_length))
    assert H * H == seq_length, f'H={H}, seq_length={seq_length}'
    return self.expand(local_2d_pattern(H, H, self.config.num_local_blocks))