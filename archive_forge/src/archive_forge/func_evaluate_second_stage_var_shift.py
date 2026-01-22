from pyomo.core.base import Objective, ConstraintList, Var, Constraint, Block
from pyomo.opt.results import TerminationCondition
from pyomo.contrib.pyros import master_problem_methods, separation_problem_methods
from pyomo.contrib.pyros.solve_data import SeparationProblemData, MasterResult
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import get_main_elapsed_time, coefficient_matching
from pyomo.core.base import value
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.var import _VarData as VarData
from itertools import chain
from pyomo.common.dependencies import numpy as np
def evaluate_second_stage_var_shift(current_master_nom_ssv_vals, previous_master_nom_ssv_vals, first_iter_master_nom_ssv_vals):
    """
    Evaluate second-stage variable "shift": the maximum relative
    difference between second-stage variable values from the current
    and previous master iterations as evaluated subject to the
    nominal uncertain parameter realization.

    Parameters
    ----------
    current_master_nom_ssv_vals : ComponentMap
        Second-stage variable values from the current master
        iteration, evaluated subject to the nominal uncertain
        parameter realization.
    previous_master_nom_ssv_vals : ComponentMap
        Second-stage variable values from the previous master
        iteration, evaluated subject to the nominal uncertain
        parameter realization.
    first_iter_master_nom_ssv_vals : ComponentMap
        Second-stage variable values from the first master
        iteration, evaluated subject to the nominal uncertain
        parameter realization.

    Returns
    -------
    None
        Returned only if `current_master_nom_ssv_vals` is empty,
        which should occur only if the problem has no second-stage
        variables.
    float
        The maximum relative difference.
        Returned only if `current_master_nom_ssv_vals` is not empty.
    """
    if not current_master_nom_ssv_vals:
        return None
    else:
        return max((abs(current_master_nom_ssv_vals[ssv] - previous_master_nom_ssv_vals[ssv]) / max((abs(first_iter_master_nom_ssv_vals[ssv]), 1)) for ssv in previous_master_nom_ssv_vals))