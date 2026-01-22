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
def perform_separation_loop(model_data, config, solve_globally):
    """
    Loop through, and solve, PyROS separation problems to
    desired optimality condition.

    Parameters
    ----------
    model_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.
    solve_globally : bool
        True to solve separation problems globally,
        False to solve separation problems locally.

    Returns
    -------
    pyros.solve_data.SeparationLoopResults
        Separation problem solve results.
    """
    all_performance_constraints = model_data.separation_model.util.performance_constraints
    if not all_performance_constraints:
        return SeparationLoopResults(solver_call_results=ComponentMap(), solved_globally=solve_globally, worst_case_perf_con=None)
    model_data.nom_perf_con_violations = evaluate_violations_by_nominal_master(model_data=model_data, performance_cons=all_performance_constraints)
    sorted_priority_groups = group_performance_constraints_by_priority(model_data, config)
    uncertainty_set_is_discrete = config.uncertainty_set.geometry == Geometry.DISCRETE_SCENARIOS
    if uncertainty_set_is_discrete:
        all_scenarios_exhausted = len(model_data.idxs_of_master_scenarios) == len(config.uncertainty_set.scenarios)
        if all_scenarios_exhausted:
            return SeparationLoopResults(solver_call_results=ComponentMap(), solved_globally=solve_globally, worst_case_perf_con=None)
        perf_con_to_maximize = sorted_priority_groups[max(sorted_priority_groups.keys())][0]
        discrete_sep_results = discrete_solve(model_data=model_data, config=config, solve_globally=solve_globally, perf_con_to_maximize=perf_con_to_maximize, perf_cons_to_evaluate=all_performance_constraints)
        termination_not_ok = discrete_sep_results.time_out or discrete_sep_results.subsolver_error
        if termination_not_ok:
            single_solver_call_res = ComponentMap()
            results_list = [res for solve_call_results in discrete_sep_results.solver_call_results.values() for res in solve_call_results.results_list]
            single_solver_call_res[perf_con_to_maximize] = SeparationSolveCallResults(solved_globally=solve_globally, results_list=results_list, time_out=discrete_sep_results.time_out, subsolver_error=discrete_sep_results.subsolver_error)
            return SeparationLoopResults(solver_call_results=single_solver_call_res, solved_globally=solve_globally, worst_case_perf_con=None)
    all_solve_call_results = ComponentMap()
    priority_groups_enum = enumerate(sorted_priority_groups.items())
    for group_idx, (priority, perf_constraints) in priority_groups_enum:
        priority_group_solve_call_results = ComponentMap()
        for idx, perf_con in enumerate(perf_constraints):
            solve_adverb = 'Globally' if solve_globally else 'Locally'
            config.progress_logger.debug(f'{solve_adverb} separating performance constraint {get_con_name_repr(model_data.separation_model, perf_con)} (priority {priority}, priority group {group_idx + 1} of {len(sorted_priority_groups)}, constraint {idx + 1} of {len(perf_constraints)} in priority group, {len(all_solve_call_results) + idx + 1} of {len(all_performance_constraints)} total)')
            if uncertainty_set_is_discrete:
                solve_call_results = get_worst_discrete_separation_solution(performance_constraint=perf_con, model_data=model_data, config=config, perf_cons_to_evaluate=all_performance_constraints, discrete_solve_results=discrete_sep_results)
            else:
                solve_call_results = solver_call_separation(model_data=model_data, config=config, solve_globally=solve_globally, perf_con_to_maximize=perf_con, perf_cons_to_evaluate=all_performance_constraints)
            priority_group_solve_call_results[perf_con] = solve_call_results
            termination_not_ok = solve_call_results.time_out or solve_call_results.subsolver_error
            if termination_not_ok:
                all_solve_call_results.update(priority_group_solve_call_results)
                return SeparationLoopResults(solver_call_results=all_solve_call_results, solved_globally=solve_globally, worst_case_perf_con=None)
        all_solve_call_results.update(priority_group_solve_call_results)
        worst_case_perf_con = get_argmax_sum_violations(solver_call_results_map=all_solve_call_results, perf_cons_to_evaluate=perf_constraints)
        if worst_case_perf_con is not None:
            worst_case_res = all_solve_call_results[worst_case_perf_con]
            if uncertainty_set_is_discrete:
                model_data.idxs_of_master_scenarios.append(worst_case_res.discrete_set_scenario_index)
            violated_con_names = '\n '.join((get_con_name_repr(model_data.separation_model, con) for con, res in all_solve_call_results.items() if res.found_violation))
            config.progress_logger.debug(f'Violated constraints:\n {violated_con_names} ')
            config.progress_logger.debug(f'Worst-case constraint: {get_con_name_repr(model_data.separation_model, worst_case_perf_con)} under realization {worst_case_res.violating_param_realization}.')
            config.progress_logger.debug(f'Maximal scaled violation {worst_case_res.scaled_violations[worst_case_perf_con]} from this constraint exceeds the robust feasibility tolerance {config.robust_feasibility_tolerance}')
            break
        else:
            config.progress_logger.debug('No violated performance constraints found.')
    return SeparationLoopResults(solver_call_results=all_solve_call_results, solved_globally=solve_globally, worst_case_perf_con=worst_case_perf_con)