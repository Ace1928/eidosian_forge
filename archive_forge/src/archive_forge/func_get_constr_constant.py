import copy
import numpy as np
from scipy.signal import fftconvolve
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
def get_constr_constant(constraints):
    """Returns the constant term for the constraints matrix.

    Parameters
    ----------
    constraints : list
        The constraints that form the matrix.

    Returns
    -------
    NumPy NDArray
        The constant term as a flattened vector.
    """
    constants = [get_constant(c.expr) for c in constraints]
    return np.hstack(constants)