import cupy
from cupy._core import core
def isintlike(x):
    try:
        return bool(int(x) == x)
    except (TypeError, ValueError):
        return False