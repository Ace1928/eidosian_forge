import math
import warnings
from itertools import combinations_with_replacement
import cupy as cp
def kernel_matrix(x, kernel_func, out):
    """Evaluate RBFs, with centers at `x`, at `x`."""
    delta = x[None, :, :] - x[:, None, :]
    out[...] = kernel_func(cp.linalg.norm(delta, axis=-1))