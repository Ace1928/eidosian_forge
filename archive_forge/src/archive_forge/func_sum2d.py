import math
import numpy as np
from numba import jit
def sum2d(s, e):
    c = 0
    for i in range(s, e):
        for j in range(s, e):
            c += i * j
    return c