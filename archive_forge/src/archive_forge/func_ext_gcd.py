from math import gcd
import numpy as np
from numpy.linalg import norm, solve
from ase.build import bulk
def ext_gcd(a, b):
    if b == 0:
        return (1, 0)
    elif a % b == 0:
        return (0, 1)
    else:
        x, y = ext_gcd(b, a % b)
        return (y, x - y * (a // b))