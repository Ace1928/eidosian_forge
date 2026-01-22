import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_loggam(x):
    a = [0.08333333333333333, -0.002777777777777778, 0.0007936507936507937, -0.0005952380952380952, 0.0008417508417508418, -0.001917526917526918, 0.00641025641025641, -0.02955065359477124, 0.1796443723688307, -1.3924322169059]
    if x == 1.0 or x == 2.0:
        return 0.0
    elif x < 7.0:
        n = int(7 - x)
    else:
        n = 0
    x0 = x + n
    x2 = 1.0 / x0 * (1.0 / x0)
    lg2pi = 1.8378770664093453
    gl0 = a[9]
    for k in range(0, 9):
        gl0 *= x2
        gl0 += a[8 - k]
    gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * np.log(x0) - x0
    if x < 7.0:
        for k in range(1, n + 1):
            gl = gl - np.log(x0 - 1.0)
            x0 = x0 - 1.0
    return gl