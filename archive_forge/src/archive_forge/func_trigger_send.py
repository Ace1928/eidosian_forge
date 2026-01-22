import itertools
from typing import List, Optional, Set, Tuple, cast
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
@triton.jit
def trigger_send(seq_num, blocks_done_counters, write_counters, other_rank, my_rank, num_stripes, stripe, num_blocks_3d, _wait, do_write):
    if (_wait and do_write) and other_rank != my_rank:
        num_blocks_done = tl.atomic_add(blocks_done_counters + other_rank + tl.arange(0, 1), 1, sem='acq_rel') + 1
        tl.atomic_xchg(write_counters + other_rank * num_stripes + stripe + tl.arange(0, 1), seq_num, mask=num_blocks_done == num_blocks_3d, sem='release')