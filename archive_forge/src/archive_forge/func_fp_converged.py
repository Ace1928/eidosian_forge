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
def fp_converged(working_model, mip_model, proj_zero_tolerance, discrete_only=True):
    """Calculates the euclidean norm between the discrete variables in the MIP and NLP models.

    Parameters
    ----------
    working_model : Pyomo model
        The working model(original model).
    mip_model : Pyomo model
        The mip model.
    proj_zero_tolerance : Float
        The projection zero tolerance of Feasibility Pump.
    discrete_only : bool, optional
        Whether to only optimize on distance between the discrete variables, by default True.

    Returns
    -------
    distance : float
        The euclidean norm between the discrete variables in the MIP and NLP models.
    """
    distance = max(((nlp_var.value - milp_var.value) ** 2 for nlp_var, milp_var in zip(working_model.MindtPy_utils.variable_list, mip_model.MindtPy_utils.variable_list) if not discrete_only or milp_var.is_integer()))
    return distance <= proj_zero_tolerance