import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_binomial_inversion(bitgen, n, p):
    q = 1.0 - p
    qn = np.exp(n * np.log(q))
    _np = n * p
    bound = min(n, _np + 10.0 * np.sqrt(_np * q + 1))
    X = 0
    px = qn
    U = next_double(bitgen)
    while U > px:
        X = X + 1
        if X > bound:
            X = 0
            px = qn
            U = next_double(bitgen)
        else:
            U -= px
            px = (n - X + 1) * p * px / (X * q)
    return X