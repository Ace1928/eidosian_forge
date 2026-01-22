import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_wald(bitgen, mean, scale):
    mu_2l = mean / (2 * scale)
    Y = random_standard_normal(bitgen)
    Y = mean * Y * Y
    X = mean + mu_2l * (Y - np.sqrt(4 * scale * Y + Y * Y))
    U = next_double(bitgen)
    if U <= mean / (mean + X):
        return X
    else:
        return mean * mean / X