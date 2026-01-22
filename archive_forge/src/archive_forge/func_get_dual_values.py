from typing import Any, Dict
import numpy as np
import cvxpy.interface as intf
def get_dual_values(result_vec, parse_func, constraints) -> Dict[Any, Any]:
    """Gets the values of the dual variables.

    Parameters
    ----------
    result_vec : array_like
        A vector containing the dual variable values.
    parse_func : function
        A function that extracts a dual value from the result vector
        for a particular constraint. The function should accept
        three arguments: the result vector, an offset, and a
        constraint, in that order. An example of a parse_func is
        extract_dual_values, defined in this module. Some solvers
        may need to implement their own parse functions.
    constraints : list
        A list of the constraints in the problem.

    Returns
    -------
       A map of constraint id to dual variable value.
    """
    dual_vars = {}
    offset = 0
    for constr in constraints:
        dual_vars[constr.id], offset = parse_func(result_vec, offset, constr)
    return dual_vars