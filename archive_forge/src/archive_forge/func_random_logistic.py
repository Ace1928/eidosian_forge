import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_logistic(bitgen, loc, scale):
    U = next_double(bitgen)
    while U <= 0.0:
        U = next_double(bitgen)
    return loc + scale * np.log(U / (1.0 - U))