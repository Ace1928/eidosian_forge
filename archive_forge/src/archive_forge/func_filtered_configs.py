import functools
import logging
from typing import cast, List, Tuple
import sympy
import torch
from torch._inductor.select_algorithm import realize_inputs
from torch._inductor.virtualized import V
from ..utils import ceildiv as cdiv, next_power_of_2
def filtered_configs(m: int, n: int, k: int, configs: List[Tuple[int, int, int, int, int]], has_int8_tensor=False):
    """Heuristic to shrink configs when they are bigger than the input size"""
    min_block_size = 32 if has_int8_tensor else 16
    m = max(next_power_of_2(V.graph.sizevars.size_hint(m, fallback=torch._inductor.config.unbacked_symint_fallback)), min_block_size)
    n = max(next_power_of_2(V.graph.sizevars.size_hint(n, fallback=torch._inductor.config.unbacked_symint_fallback)), min_block_size)
    k = max(next_power_of_2(V.graph.sizevars.size_hint(k, fallback=torch._inductor.config.unbacked_symint_fallback)), min_block_size)
    used = set()
    for block_m, block_n, block_k, num_stages, num_warps in configs:
        block_m = max(min(block_m, m), min_block_size)
        block_n = max(min(block_n, n), min_block_size)
        block_k = max(min(block_k, k), min_block_size)
        num_warps = min(num_warps, block_m * block_n // 256)
        if (block_m, block_n, block_k, num_stages, num_warps) not in used:
            used.add((block_m, block_n, block_k, num_stages, num_warps))
            yield triton_config(BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, num_stages=num_stages, num_warps=num_warps)