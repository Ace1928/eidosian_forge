import cupy
from cupy._core import core
def isscalarlike(x):
    return cupy.isscalar(x) or (isdense(x) and x.ndim == 0)