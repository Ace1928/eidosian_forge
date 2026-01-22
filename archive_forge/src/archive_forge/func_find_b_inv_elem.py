import operator
from math import prod
import numpy as np
from scipy._lib._util import normalize_axis_index
from scipy.linalg import (get_lapack_funcs, LinAlgError,
from scipy.optimize import minimize_scalar
from . import _bspl
from . import _fitpack_impl
from scipy.sparse import csr_array
from scipy.special import poch
from itertools import combinations
def find_b_inv_elem(i, j, U, D, B):
    rng = min(3, n - i - 1)
    rng_sum = 0.0
    if j == 0:
        for k in range(1, rng + 1):
            rng_sum -= U[-k - 1, i + k] * B[-k - 1, i + k]
        rng_sum += D[i]
        B[-1, i] = rng_sum
    else:
        for k in range(1, rng + 1):
            diag = abs(k - j)
            ind = i + min(k, j)
            rng_sum -= U[-k - 1, i + k] * B[-diag - 1, ind + diag]
        B[-j - 1, i + j] = rng_sum