import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_laplace(bitgen, loc, scale):
    U = next_double(bitgen)
    while U <= 0:
        U = next_double(bitgen)
    if U >= 0.5:
        U = loc - scale * np.log(2.0 - U - U)
    elif U > 0.0:
        U = loc + scale * np.log(U + U)
    return U