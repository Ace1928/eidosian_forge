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
def make_separation_objective_functions(model, config):
    """
    Inequality constraints referencing control variables, state variables, or uncertain parameters
    must be separated against in separation problem.
    """
    performance_constraints = []
    for c in model.component_data_objects(Constraint, active=True, descend_into=True):
        _vars = ComponentSet(identify_variables(expr=c.expr))
        uncertain_params_in_expr = list((v for v in model.util.uncertain_param_vars.values() if v in _vars))
        state_vars_in_expr = list((v for v in model.util.state_vars if v in _vars))
        second_stage_variables_in_expr = list((v for v in model.util.second_stage_variables if v in _vars))
        if not c.equality and (uncertain_params_in_expr or state_vars_in_expr or second_stage_variables_in_expr):
            performance_constraints.append(c)
        elif not c.equality and (not (uncertain_params_in_expr or state_vars_in_expr or second_stage_variables_in_expr)):
            c.deactivate()
    model.util.performance_constraints = performance_constraints
    model.util.separation_objectives = []
    map_obj_to_constr = ComponentMap()
    for idx, c in enumerate(performance_constraints):
        c.deactivate()
        if c.upper is not None:
            obj = Objective(expr=c.body - c.upper, sense=maximize)
            map_obj_to_constr[c] = obj
            model.add_component('separation_obj_' + str(idx), obj)
            model.util.separation_objectives.append(obj)
        elif c.lower is not None:
            raise ValueError('All inequality constraints in model must be in standard form (<= RHS)')
    model.util.map_obj_to_constr = map_obj_to_constr
    for obj in model.util.separation_objectives:
        obj.deactivate()
    return