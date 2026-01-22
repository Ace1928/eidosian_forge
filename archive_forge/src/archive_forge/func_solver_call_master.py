from pyomo.core.base import (
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverResults
from pyomo.core.expr import value
from pyomo.core.base.set_types import NonNegativeIntegers, NonNegativeReals
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, MasterResult
from pyomo.opt.results import check_optimal_termination
from pyomo.core.expr.visitor import replace_expressions, identify_variables
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core import TransformationFactory
import itertools as it
import os
from copy import deepcopy
from pyomo.common.errors import ApplicationError
from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import TIC_TOC_SOLVE_TIME_ATTR, enforce_dr_degree
def solver_call_master(model_data, config, solver, solve_data):
    """
    Invoke subsolver(s) on PyROS master problem.

    Parameters
    ----------
    model_data : MasterProblemData
        Container for current master problem and related data.
    config : ConfigDict
        PyROS solver settings.
    solver : solver type
        Primary subordinate optimizer with which to solve
        the master problem. This may be a local or global
        NLP solver.
    solve_data : MasterResult
        Master problem results object. May be empty or contain
        master feasibility problem results.

    Returns
    -------
    master_soln : MasterResult
        Master problem results object, containing master
        model and subsolver results.
    """
    nlp_model = model_data.master_model
    master_soln = solve_data
    solver_term_cond_dict = {}
    if config.solve_master_globally:
        solvers = [solver] + config.backup_global_solvers
    else:
        solvers = [solver] + config.backup_local_solvers
    higher_order_decision_rule_efficiency(model_data=model_data, config=config)
    solve_mode = 'global' if config.solve_master_globally else 'local'
    config.progress_logger.debug('Solving master problem')
    timer = TicTocTimer()
    for idx, opt in enumerate(solvers):
        if idx > 0:
            config.progress_logger.warning(f'Invoking backup solver {opt!r} (solver {idx + 1} of {len(solvers)}) for master problem of iteration {model_data.iteration}.')
        orig_setting, custom_setting_present = adjust_solver_time_settings(model_data.timing, opt, config)
        model_data.timing.start_timer('main.master')
        timer.tic(msg=None)
        try:
            results = opt.solve(nlp_model, tee=config.tee, load_solutions=False, symbolic_solver_labels=True)
        except ApplicationError:
            config.progress_logger.error(f'Optimizer {repr(opt)} ({idx + 1} of {len(solvers)}) encountered exception attempting to solve master problem in iteration {model_data.iteration}')
            raise
        else:
            setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, timer.toc(msg=None))
            model_data.timing.stop_timer('main.master')
        finally:
            revert_solver_max_time_adjustment(solver, orig_setting, custom_setting_present, config)
        optimal_termination = check_optimal_termination(results)
        infeasible = results.solver.termination_condition == tc.infeasible
        if optimal_termination:
            nlp_model.solutions.load_from(results)
        solver_term_cond_dict[str(opt)] = str(results.solver.termination_condition)
        master_soln.termination_condition = results.solver.termination_condition
        master_soln.pyros_termination_condition = None
        try_backup, _ = master_soln.master_subsolver_results = process_termination_condition_master_problem(config=config, results=results)
        master_soln.nominal_block = nlp_model.scenarios[0, 0]
        master_soln.results = results
        master_soln.master_model = nlp_model
        if not try_backup and (not infeasible):
            master_soln.fsv_vals = list((v.value for v in nlp_model.scenarios[0, 0].util.first_stage_variables))
            if config.objective_focus is ObjectiveType.nominal:
                master_soln.ssv_vals = list((v.value for v in nlp_model.scenarios[0, 0].util.second_stage_variables))
                master_soln.second_stage_objective = value(nlp_model.scenarios[0, 0].second_stage_objective)
            else:
                idx = max(nlp_model.scenarios.keys())[0]
                master_soln.ssv_vals = list((v.value for v in nlp_model.scenarios[idx, 0].util.second_stage_variables))
                master_soln.second_stage_objective = value(nlp_model.scenarios[idx, 0].second_stage_objective)
            master_soln.first_stage_objective = value(nlp_model.scenarios[0, 0].first_stage_objective)
            if config.objective_focus == ObjectiveType.worst_case:
                eval_obj_blk_idx = max(nlp_model.scenarios.keys(), key=lambda idx: value(nlp_model.scenarios[idx].second_stage_objective))
            else:
                eval_obj_blk_idx = (0, 0)
            eval_obj_blk = nlp_model.scenarios[eval_obj_blk_idx]
            config.progress_logger.debug(' Optimized master objective breakdown:')
            config.progress_logger.debug(f'  First-stage objective: {value(eval_obj_blk.first_stage_objective)}')
            config.progress_logger.debug(f'  Second-stage objective: {value(eval_obj_blk.second_stage_objective)}')
            master_obj = eval_obj_blk.first_stage_objective + eval_obj_blk.second_stage_objective
            config.progress_logger.debug(f'  Objective: {value(master_obj)}')
            config.progress_logger.debug(f' Termination condition: {results.solver.termination_condition}')
            config.progress_logger.debug(f' Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)}s')
            master_soln.nominal_block = nlp_model.scenarios[0, 0]
            master_soln.results = results
            master_soln.master_model = nlp_model
        elapsed = get_main_elapsed_time(model_data.timing)
        if config.time_limit:
            if elapsed >= config.time_limit:
                try_backup = False
                master_soln.master_subsolver_results = (None, pyrosTerminationCondition.time_out)
                master_soln.pyros_termination_condition = pyrosTerminationCondition.time_out
        if not try_backup:
            return master_soln
    save_dir = config.subproblem_file_directory
    serialization_msg = ''
    if save_dir and config.keepfiles:
        output_problem_path = os.path.join(save_dir, config.uncertainty_set.type + '_' + model_data.original.name + '_master_' + str(model_data.iteration) + '.bar')
        nlp_model.write(output_problem_path, io_options={'symbolic_solver_labels': True})
        serialization_msg = f' For debugging, problem has been serialized to the file {output_problem_path!r}.'
    deterministic_model_qual = ' (i.e., the deterministic model)' if model_data.iteration == 0 else ''
    deterministic_msg = f' Please ensure your deterministic model is solvable by at least one of the subordinate {solve_mode} optimizers provided.' if model_data.iteration == 0 else ''
    master_soln.pyros_termination_condition = pyrosTerminationCondition.subsolver_error
    config.progress_logger.warning(f'Could not successfully solve master problem of iteration {model_data.iteration}{deterministic_model_qual} with any of the provided subordinate {solve_mode} optimizers. (Termination statuses: {[term_cond for term_cond in solver_term_cond_dict.values()]}.){deterministic_msg}{serialization_msg}')
    return master_soln