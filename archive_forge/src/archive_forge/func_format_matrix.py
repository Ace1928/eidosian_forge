from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def format_matrix(matrix, shape=None, format='dense'):
    """Returns the matrix in the appropriate form for SWIG wrapper"""
    if format == 'dense':
        if len(shape) == 0:
            shape = (1, 1)
        elif len(shape) == 1:
            shape = shape + (1,)
        return np.reshape(matrix, shape, order='F')
    elif format == 'sparse':
        return sp.coo_matrix(matrix)
    elif format == 'scalar':
        return np.asfortranarray([[matrix]])
    else:
        raise NotImplementedError()