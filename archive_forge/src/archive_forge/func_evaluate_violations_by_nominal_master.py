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
def evaluate_violations_by_nominal_master(model_data, performance_cons):
    """
    Evaluate violation of performance constraints by
    variables in nominal block of most recent master
    problem.

    Returns
    -------
    nom_perf_con_violations : dict
        Mapping from performance constraint names
        to floats equal to violations by nominal master
        problem variables.
    """
    constraint_map_to_master = model_data.separation_model.util.map_new_constraint_list_to_original_con
    set_of_deterministic_constraints = model_data.separation_model.util.deterministic_constraints
    if hasattr(model_data.separation_model, 'epigraph_constr'):
        set_of_deterministic_constraints.add(model_data.separation_model.epigraph_constr)
    nom_perf_con_violations = {}
    for perf_con in performance_cons:
        if perf_con in set_of_deterministic_constraints:
            nom_constraint = perf_con
        else:
            nom_constraint = constraint_map_to_master[perf_con]
        nom_violation = value(model_data.master_nominal_scenario.find_component(nom_constraint))
        nom_perf_con_violations[perf_con] = nom_violation
    return nom_perf_con_violations