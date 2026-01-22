import itertools
from typing import List, Optional, Set, Tuple, cast
import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time
@triton.jit
def wait_for_recv(seq_num, wait_counters, other_rank, my_rank, stripe, num_stripes, _wait, do_wait, timeout_ns):
    if (_wait and do_wait) and other_rank != my_rank:
        wait_counter = wait_counters + other_rank * num_stripes + stripe
        start_time_ns = tl.extra.cuda.globaltimer()
        num_spins = 0
        while tl.atomic_cas(wait_counter, 0, 0) != seq_num:
            num_spins += 1
            if num_spins == NUM_SPINS_BETWEEN_TIMEOUT_CHECKS:
                if tl.extra.cuda.globaltimer() - start_time_ns > timeout_ns:
                    tl.device_assert(False, "xFormers's fused kernels for sequence parallelism timed out waiting for a peer GPU. To prevent downstream computations from operating on corrupted data, we're bringing the CUDA context down with us.")
                num_spins = 0