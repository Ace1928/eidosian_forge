from cupy import _core
from cupy._core import fusion
from cupy._math import ufunc
def round_(a, decimals=0, out=None):
    return around(a, decimals, out=out)