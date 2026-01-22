from __future__ import annotations
import numbers
import os
import numpy as np
import scipy.sparse as sp
import cvxpy.cvxcore.python.cvxcore as cvxcore
import cvxpy.settings as s
from cvxpy.lin_ops import lin_op as lo
from cvxpy.lin_ops.canon_backend import CanonBackend
def get_parameter_vector(param_size, param_id_to_col, param_id_to_size, param_id_to_value_fn, zero_offset: bool=False):
    """Returns a flattened parameter vector

    The flattened vector includes a constant offset (i.e, a 1).

    Parameters
    ----------
        param_size: The number of parameters
        param_id_to_col: A dict from parameter id to column offset
        param_id_to_size: A dict from parameter id to parameter size
        param_id_to_value_fn: A callable that returns a value for a parameter id
        zero_offset: (optional) if True, zero out the constant offset in the
                     parameter vector

    Returns
    -------
        A flattened NumPy array of parameter values, of length param_size + 1
    """
    if param_size == 0:
        return None
    param_vec = np.zeros(param_size + 1)
    for param_id, col in param_id_to_col.items():
        if param_id == lo.CONSTANT_ID:
            if not zero_offset:
                param_vec[col] = 1
        else:
            value = param_id_to_value_fn(param_id).flatten(order='F')
            size = param_id_to_size[param_id]
            param_vec[col:col + size] = value
    return param_vec