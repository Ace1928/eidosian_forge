import logging
from pyomo.common.collections import ComponentMap
from pyomo.core import (
from pyomo.repn import generate_standard_repn
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
import pyomo.core.expr as EXPR
from pyomo.opt import ProblemSense
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.util.model_size import build_model_size_report
from pyomo.common.dependencies import attempt_import
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
import math
def get_integer_solution(model, string_zero=False):
    """Extract the value of integer variables from the provided model.

    Parameters
    ----------
    model : Pyomo model
        The model to extract value of integer variables.
    string_zero : bool, optional
        Whether to store zero as string, by default False.

    Returns
    -------
    tuple
        The tuple of integer variable values.
    """
    temp = []
    for var in model.MindtPy_utils.discrete_variable_list:
        if string_zero:
            if var.value == 0:
                temp.append(str(var.value))
            else:
                temp.append(int(round(var.value)))
        else:
            temp.append(int(round(var.value)))
    return tuple(temp)