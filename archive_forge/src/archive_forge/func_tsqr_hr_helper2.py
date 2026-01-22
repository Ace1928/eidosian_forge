import numpy as np
import ray
import ray.experimental.array.remote as ra
from . import core
@ray.remote
def tsqr_hr_helper2(s, r_temp):
    s_full = np.diag(s)
    return np.dot(s_full, r_temp)