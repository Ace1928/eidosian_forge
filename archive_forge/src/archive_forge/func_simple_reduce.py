import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo
def simple_reduce(self, rank):
    """
        Reduce the rank of the matrix by dropping oldest vectors.
        """
    if self.collapsed is not None:
        return
    assert rank > 0
    while len(self.cs) > rank:
        del self.cs[0]
        del self.ds[0]