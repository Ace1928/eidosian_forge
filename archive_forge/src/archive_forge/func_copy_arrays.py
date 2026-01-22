import math
import numpy as np
from numba import jit
def copy_arrays(a, b):
    for i in range(a.shape[0]):
        b[i] = a[i]