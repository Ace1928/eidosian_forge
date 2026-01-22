from ctypes import *
import sys
import numpy as np
from numba import _helperlib
def use_c_vcube(x):
    out = np.empty_like(x)
    c_vcube(x.size, x.ctypes, out.ctypes)
    return out