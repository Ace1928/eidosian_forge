import math
from bisect import bisect
import sys
from .backend import (MPZ, MPZ_TYPE, MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_FIVE,
from .libintmath import (giant_steps,
def str_to_man_exp(x, base=10):
    """Helper function for from_str."""
    x = x.lower().rstrip('l')
    float(x)
    parts = x.split('e')
    if len(parts) == 1:
        exp = 0
    else:
        x = parts[0]
        exp = int(parts[1])
    parts = x.split('.')
    if len(parts) == 2:
        a, b = (parts[0], parts[1].rstrip('0'))
        exp -= len(b)
        x = a + b
    x = MPZ(int(x, base))
    return (x, exp)