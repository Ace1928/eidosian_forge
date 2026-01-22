import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_weibull(bitgen, a):
    if a == 0.0:
        return 0.0
    return pow(random_standard_exponential(bitgen), 1.0 / a)