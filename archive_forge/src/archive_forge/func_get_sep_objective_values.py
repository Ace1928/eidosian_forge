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
def get_sep_objective_values(model_data, config, perf_cons):
    """
    Evaluate performance constraint functions at current
    separation solution.

    Parameters
    ----------
    model_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.
    perf_cons : list of Constraint
        Performance constraints to be evaluated.

    Returns
    -------
    violations : ComponentMap
        Mapping from performance constraints to violation values.
    """
    con_to_obj_map = model_data.separation_model.util.map_obj_to_constr
    violations = ComponentMap()
    for perf_con in perf_cons:
        obj = con_to_obj_map[perf_con]
        try:
            violations[perf_con] = value(obj.expr)
        except ValueError:
            for v in model_data.separation_model.util.first_stage_variables:
                config.progress_logger.info(v.name + ' ' + str(v.value))
            for v in model_data.separation_model.util.second_stage_variables:
                config.progress_logger.info(v.name + ' ' + str(v.value))
            raise ArithmeticError(f'Evaluation of performance constraint {perf_con.name} (separation objective {obj.name}) led to a math domain error. Does the performance constraint expression contain log(x) or 1/x functions or others with tricky domains?')
    return violations