import math
from numba import jit
from numba.core import types
from math import sqrt
import numpy as np
import numpy.random as nprand
@jit(nopython=True)
def get_renamed_module(x):
    nprand.seed(42)
    return (np.cos(x), nprand.random())