import cupy
from cupy._core import core
def isshape(x):
    if not isinstance(x, tuple) or len(x) != 2:
        return False
    m, n = x
    if isinstance(n, tuple):
        return False
    return isintlike(m) and isintlike(n)