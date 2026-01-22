import math
from numba import (config, cuda, float32, float64, uint32, int64, uint64,
import numpy as np
@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def init_xoroshiro128p_states_cpu(states, seed, subsequence_start):
    n = states.shape[0]
    seed = uint64(seed)
    subsequence_start = uint64(subsequence_start)
    if n >= 1:
        init_xoroshiro128p_state(states, 0, seed)
        for _ in range(subsequence_start):
            xoroshiro128p_jump(states, 0)
        for i in range(1, n):
            states[i] = states[i - 1]
            xoroshiro128p_jump(states, i)