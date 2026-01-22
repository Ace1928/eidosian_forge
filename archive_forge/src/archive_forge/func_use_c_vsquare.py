from ctypes import *
import sys
import numpy as np
from numba import _helperlib
def use_c_vsquare(x):
    out = np.empty_like(x)
    c_vsquare(x.size, x.ctypes, out.ctypes)
    return out