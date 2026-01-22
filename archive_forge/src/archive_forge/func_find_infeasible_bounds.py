from pyomo.core import Constraint, Var, value
from math import fabs
import logging
from pyomo.common import deprecated
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.blockutil import log_model_constraints
def find_infeasible_bounds(m, tol=1e-06):
    """Find variables whose values are outside their bounds

    Uses the current model state. Variables with no values are returned
    as if they were infeasible.

    Parameters
    ----------
    m: Block
        Pyomo block or model to check

    tol: float
        absolute feasibility tolerance

    Yields
    ------
    var: VarData
        The variable that is outside its bounds

    infeasible: int
        A bitmask indicating which bound was infeasible (1 for the lower
        bound or 2 for the upper bound; 4 indicates the variable had no
        value or a bound was undefined)

    """
    for var in m.component_data_objects(ctype=Var, descend_into=True):
        val = var.value
        infeasible = _check_infeasible(var, val, tol)
        if infeasible:
            yield (var, infeasible)