from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.objective import Objective, maximize, value
from pyomo.core.base import Var, Param
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pyros.util import ObjectiveType, get_time_from_solver
from pyomo.contrib.pyros.solve_data import (
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import get_main_elapsed_time, is_certain_parameter
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.common.errors import ApplicationError
from pyomo.contrib.pyros.util import ABS_CON_CHECK_FEAS_TOL
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import (
import os
from copy import deepcopy
from itertools import product
def initialize_separation(perf_con_to_maximize, model_data, config):
    """
    Initialize separation problem variables, and fix all first-stage
    variables to their corresponding values from most recent
    master problem solution.

    Parameters
    ----------
    perf_con_to_maximize : ConstraintData
        Performance constraint whose violation is to be maximized
        for the separation problem of interest.
    model_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.

    Note
    ----
    If a static DR policy is used, then all second-stage variables
    are fixed and the decision rule equations are deactivated.

    The point to which the separation model is initialized should,
    in general, be feasible, provided the set does not have a
    discrete geometry (as there is no master model block corresponding
    to any of the remaining discrete scenarios against which we
    separate).

    This method assumes that the master model has only one block
    per iteration.
    """

    def eval_master_violation(block_idx):
        """
        Evaluate violation of `perf_con` by variables of
        specified master block.
        """
        new_con_map = model_data.separation_model.util.map_new_constraint_list_to_original_con
        in_new_cons = perf_con_to_maximize in new_con_map
        if in_new_cons:
            sep_con = new_con_map[perf_con_to_maximize]
        else:
            sep_con = perf_con_to_maximize
        master_con = model_data.master_model.scenarios[block_idx, 0].find_component(sep_con)
        return value(master_con)
    block_num = max(range(model_data.iteration + 1), key=eval_master_violation)
    master_blk = model_data.master_model.scenarios[block_num, 0]
    master_blks = list(model_data.master_model.scenarios.values())
    fsv_set = ComponentSet(master_blk.util.first_stage_variables)
    sep_model = model_data.separation_model

    def get_parent_master_blk(var):
        """
        Determine the master model scenario block of which
        a given variable is a child component (or descendant).
        """
        parent = var.parent_block()
        while parent not in master_blks:
            parent = parent.parent_block()
        return parent
    for master_var in master_blk.component_data_objects(Var, active=True):
        parent_master_blk = get_parent_master_blk(master_var)
        sep_var_name = master_var.getname(relative_to=parent_master_blk, fully_qualified=True)
        sep_var = sep_model.find_component(sep_var_name)
        sep_var.set_value(value(master_var, exception=False))
        if master_var in fsv_set:
            sep_var.fix()
    if config.uncertainty_set.geometry != Geometry.DISCRETE_SCENARIOS:
        param_vars = sep_model.util.uncertain_param_vars
        latest_param_values = model_data.points_added_to_master[block_num]
        for param_var, val in zip(param_vars.values(), latest_param_values):
            param_var.set_value(val)
    for c in model_data.separation_model.util.second_stage_variables:
        if config.decision_rule_order != 0:
            c.unfix()
        else:
            c.fix()
    if config.decision_rule_order == 0:
        for v in model_data.separation_model.util.decision_rule_eqns:
            v.deactivate()
        for v in model_data.separation_model.util.decision_rule_vars:
            v.fix()
    if any((c.active for c in model_data.separation_model.util.h_x_q_constraints)):
        raise AttributeError('All h(x,q) type constraints must be deactivated in separation.')
    tol = ABS_CON_CHECK_FEAS_TOL
    perf_con_name_repr = get_con_name_repr(separation_model=model_data.separation_model, con=perf_con_to_maximize, with_orig_name=True, with_obj_name=True)
    uncertainty_set_is_discrete = config.uncertainty_set.geometry is Geometry.DISCRETE_SCENARIOS
    for con in sep_model.component_data_objects(Constraint, active=True):
        lslack, uslack = (con.lslack(), con.uslack())
        if (lslack < -tol or uslack < -tol) and (not uncertainty_set_is_discrete):
            con_name_repr = get_con_name_repr(separation_model=model_data.separation_model, con=con, with_orig_name=True, with_obj_name=False)
            config.progress_logger.debug(f'Initial point for separation of performance constraint {perf_con_name_repr} violates the model constraint {con_name_repr} by more than {tol}. (lslack={con.lslack()}, uslack={con.uslack()})')