import copy
import numpy as np
from scipy.signal import fftconvolve
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
Prunes constant branches from the expression.

    Parameters
    ----------
    lin_op : LinOp
        The root linear operator.

    Returns
    -------
    bool
        Were all the expression's arguments pruned?
    