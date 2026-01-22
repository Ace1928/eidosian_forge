import pyomo.environ as pyo
import math
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentSet
from pyomo.core.base.var import _GeneralVarData
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
import logging

    This function logs potentially poorly scaled parts of the model.
    It requires that all variables be bounded.

    It is important to note that this check is neither necessary nor sufficient
    to ensure a well-scaled model. However, it is a useful tool to help identify
    problematic parts of a model.

    This function uses symbolic differentiation and interval arithmetic
    to compute bounds on each entry in the jacobian of the constraints.

    Note that logging has to be turned on to get the output

    Parameters
    ----------
    m: _BlockData
        The pyomo model or block
    too_large: float
        Values above too_large will generate a log entry
    too_small: float
        Coefficients below too_small will generate a log entry

    Returns
    -------
    success: bool
        Returns False if any potentially poorly scaled components were found
    