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
def generate_norm1_objective_function(model, setpoint_model, discrete_only=False):
    """This function generates objective (PF-OA main problem) for minimum
    Norm1 distance to setpoint_model.

    Norm1 distance of (x,y) = \\sum_i |x_i - y_i|.

    Parameters
    ----------
    model : Pyomo model
        The model that needs new objective function.
    setpoint_model : Pyomo model
        The model that provides the base point for us to calculate the distance.
    discrete_only : bool, optional
        Whether to only optimize on distance between the discrete
        variables, by default False.

    Returns
    -------
    Objective
        The norm1 objective function.

    """
    var_filter = (lambda v: v.is_integer()) if discrete_only else lambda v: 'MindtPy_utils.objective_value' not in v.name and 'MindtPy_utils.feas_opt.slack_var' not in v.name
    model_vars = list(filter(var_filter, model.MindtPy_utils.variable_list))
    setpoint_vars = list(filter(var_filter, setpoint_model.MindtPy_utils.variable_list))
    assert len(model_vars) == len(setpoint_vars), 'Trying to generate Norm1 objective function for models with different number of variables'
    model.MindtPy_utils.del_component('L1_obj')
    obj_block = model.MindtPy_utils.L1_obj = Block()
    obj_block.L1_obj_idx = RangeSet(len(model_vars))
    obj_block.L1_obj_var = Var(obj_block.L1_obj_idx, domain=Reals, bounds=(0, None))
    obj_block.abs_reform = ConstraintList()
    for idx, v_model, v_setpoint in zip(obj_block.L1_obj_idx, model_vars, setpoint_vars):
        obj_block.abs_reform.add(expr=v_model - v_setpoint.value >= -obj_block.L1_obj_var[idx])
        obj_block.abs_reform.add(expr=v_model - v_setpoint.value <= obj_block.L1_obj_var[idx])
    return Objective(expr=sum((obj_block.L1_obj_var[idx] for idx in obj_block.L1_obj_idx)))