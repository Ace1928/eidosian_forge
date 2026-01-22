import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def iamax(x, out=None):
    """Finds the (smallest) index of the element with the maximum magnitude.

    Note: The result index is 1-based index (not 0-based index).
    """
    return _iamaxmin(x, out, 'amax')