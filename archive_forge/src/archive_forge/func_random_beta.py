import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_beta(bitgen, a, b):
    if a <= 1.0 and b <= 1.0:
        while 1:
            U = next_double(bitgen)
            V = next_double(bitgen)
            X = pow(U, 1.0 / a)
            Y = pow(V, 1.0 / b)
            XpY = X + Y
            if XpY <= 1.0 and XpY > 0.0:
                if X + Y > 0:
                    return X / XpY
                else:
                    logX = np.log(U) / a
                    logY = np.log(V) / b
                    logM = min(logX, logY)
                    logX -= logM
                    logY -= logM
                    return np.exp(logX - np.log(np.exp(logX) + np.exp(logY)))
    else:
        Ga = random_standard_gamma(bitgen, a)
        Gb = random_standard_gamma(bitgen, b)
        return Ga / (Ga + Gb)