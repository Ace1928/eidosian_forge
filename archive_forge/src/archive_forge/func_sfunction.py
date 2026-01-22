import numpy
from numpy import asarray_chkfinite, single, asarray, array
from numpy.linalg import norm
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs
from ._decomp import eigvals
def sfunction(x):
    return abs(x) > 1.0