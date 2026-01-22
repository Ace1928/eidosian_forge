from typing import Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.utilities as u
from cvxpy.lin_ops.lin_constraints import LinEqConstr, LinLeqConstr
def check_param_val(param):
    """Wrapper on accessing a parameter.

    Parameters
    ----------
    param : Parameter
        The parameter whose value is being accessed.

    Returns
    -------
    The numerical value of the parameter.

    Raises
    ------
    ValueError
        Raises error if parameter value is None.
    """
    val = param.value
    if val is None:
        raise ValueError('Problem has missing parameter value.')
    else:
        return val