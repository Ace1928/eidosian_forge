import numpy as np
import numba as nb
from numpy.random import PCG64
from timeit import timeit
@nb.jit(nopython=True)
def bounded_uints(lb, ub, n, state):
    out = np.empty(n, dtype=np.uint32)
    for i in range(n):
        out[i] = bounded_uint(lb, ub, state)