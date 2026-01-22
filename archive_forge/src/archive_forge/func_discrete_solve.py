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
def discrete_solve(model_data, config, solve_globally, perf_con_to_maximize, perf_cons_to_evaluate):
    """
    Obtain separation problem solution for each scenario
    of the uncertainty set not already added to the most
    recent master problem.

    Parameters
    ----------
    model_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.
    solver : solver type
        Primary subordinate optimizer with which to solve
        the model.
    solve_globally : bool
        Is separation problem to be solved globally.
    perf_con_to_maximize : Constraint
        Performance constraint for which to solve separation
        problem.
    perf_cons_to_evaluate : list of Constraint
        Performance constraints whose expressions are to be
        evaluated at the each of separation problem solutions
        obtained.

    Returns
    -------
    discrete_separation_results : DiscreteSeparationSolveCallResults
        Separation solver call results on performance constraint
        of interest for every scenario considered.

    Notes
    -----
    Since we assume that models passed to PyROS are such that the DOF
    variables and uncertain parameter values uniquely define the state
    variables, this method need be only be invoked once per separation
    loop. Subject to our assumption, the choice of objective
    (``perf_con_to_maximize``) should not affect the solutions returned
    beyond subsolver tolerances. For other performance constraints, the
    optimal separation problem solution can then be evaluated by simple
    enumeration of the solutions returned by this function, since for
    discrete uncertainty sets, the number of feasible separation
    solutions is, under our assumption, merely equal to the number
    of scenarios in the uncertainty set.
    """
    model_data.separation_model.util.uncertainty_set_constraint.deactivate()
    uncertain_param_vars = list(model_data.separation_model.util.uncertain_param_vars.values())
    master_scenario_idxs = model_data.idxs_of_master_scenarios
    scenario_idxs_to_separate = [idx for idx, _ in enumerate(config.uncertainty_set.scenarios) if idx not in master_scenario_idxs]
    solve_call_results_dict = {}
    for scenario_idx in scenario_idxs_to_separate:
        scenario = config.uncertainty_set.scenarios[scenario_idx]
        for param, coord_val in zip(uncertain_param_vars, scenario):
            param.fix(coord_val)
        solve_call_results = solver_call_separation(model_data=model_data, config=config, solve_globally=solve_globally, perf_con_to_maximize=perf_con_to_maximize, perf_cons_to_evaluate=perf_cons_to_evaluate)
        solve_call_results.discrete_set_scenario_index = scenario_idx
        solve_call_results_dict[scenario_idx] = solve_call_results
        termination_not_ok = solve_call_results.subsolver_error or solve_call_results.time_out
        if termination_not_ok:
            break
    return DiscreteSeparationSolveCallResults(solved_globally=solve_globally, solver_call_results=solve_call_results_dict, performance_constraint=perf_con_to_maximize)