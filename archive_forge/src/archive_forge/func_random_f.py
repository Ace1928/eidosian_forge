import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_f(bitgen, dfnum, dfden):
    return random_chisquare(bitgen, dfnum) * dfden / (random_chisquare(bitgen, dfden) * dfnum)