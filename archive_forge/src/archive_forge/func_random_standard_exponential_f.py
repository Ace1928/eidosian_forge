import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_standard_exponential_f(bitgen):
    while 1:
        ri = next_uint32(bitgen)
        ri >>= 1
        idx = ri & 255
        ri >>= 8
        x = float32(float32(ri) * we_float[idx])
        if ri < ke_float[idx]:
            return x
        elif idx == 0:
            return float32(ziggurat_exp_r_f - float32(np_log1pf(-next_float(bitgen))))
        elif (fe_float[idx - 1] - fe_float[idx]) * next_float(bitgen) + fe_float[idx] < float32(np.exp(float32(-x))):
            return x