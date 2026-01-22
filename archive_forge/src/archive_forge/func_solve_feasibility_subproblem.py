import math
from io import StringIO
import pyomo.core.expr as EXPR
from pyomo.repn import generate_standard_repn
import logging
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.opt import TerminationCondition as tc
from pyomo.contrib.mindtpy import __version__
from pyomo.common.dependencies import attempt_import
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.common.collections import ComponentMap, Bunch, ComponentSet
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.mindtpy.cut_generation import add_no_good_cuts
from operator import itemgetter
from pyomo.common.errors import DeveloperError
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.opt import (
from pyomo.core import (
from pyomo.contrib.gdpopt.util import (
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.contrib.mindtpy.util import (
def solve_feasibility_subproblem(self):
    """Solves a feasibility NLP if the fixed_nlp problem is infeasible.

        Returns
        -------
        feas_subproblem : Pyomo model
            Feasibility NLP from the model.
        feas_soln : SolverResults
            Results from solving the feasibility NLP.
        """
    config = self.config
    feas_subproblem = self.fixed_nlp
    MindtPy = feas_subproblem.MindtPy_utils
    MindtPy.feas_opt.activate()
    if MindtPy.component('objective_value') is not None:
        MindtPy.objective_value[:].set_value(0, skip_validation=True)
    active_obj = next(feas_subproblem.component_data_objects(Objective, active=True))
    active_obj.deactivate()
    for constr in MindtPy.nonlinear_constraint_list:
        constr.deactivate()
    MindtPy.feas_opt.activate()
    MindtPy.feas_obj.activate()
    nlp_args = dict(config.nlp_solver_args)
    update_solver_timelimit(self.feasibility_nlp_opt, config.nlp_solver, self.timing, config)
    TransformationFactory('contrib.deactivate_trivial_constraints').apply_to(feas_subproblem, tmp=True, ignore_infeasible=False, tolerance=config.constraint_tolerance)
    with SuppressInfeasibleWarning():
        try:
            with time_code(self.timing, 'feasibility subproblem'):
                feas_soln = self.feasibility_nlp_opt.solve(feas_subproblem, tee=config.nlp_solver_tee, load_solutions=config.nlp_solver != 'appsi_ipopt', **nlp_args)
                if len(feas_soln.solution) > 0:
                    feas_subproblem.solutions.load_from(feas_soln)
        except (ValueError, OverflowError) as e:
            config.logger.error(e, exc_info=True)
            for nlp_var, orig_val in zip(MindtPy.variable_list, self.initial_var_values):
                if not nlp_var.fixed and (not nlp_var.is_binary()):
                    nlp_var.set_value(orig_val, skip_validation=True)
            with time_code(self.timing, 'feasibility subproblem'):
                feas_soln = self.feasibility_nlp_opt.solve(feas_subproblem, tee=config.nlp_solver_tee, load_solutions=config.nlp_solver != 'appsi_ipopt', **nlp_args)
                if len(feas_soln.solution) > 0:
                    feas_soln.solutions.load_from(feas_soln)
    self.handle_feasibility_subproblem_tc(feas_soln.solver.termination_condition, MindtPy)
    config.logger.info(self.fixed_nlp_log_formatter.format(' ', self.nlp_iter, 'Feasibility NLP', value(feas_subproblem.MindtPy_utils.feas_obj), self.primal_bound, self.dual_bound, self.rel_gap, get_main_elapsed_time(self.timing)))
    MindtPy.feas_opt.deactivate()
    for constr in MindtPy.nonlinear_constraint_list:
        constr.activate()
    active_obj.activate()
    MindtPy.feas_obj.deactivate()
    TransformationFactory('contrib.deactivate_trivial_constraints').revert(feas_subproblem)
    return (feas_subproblem, feas_soln)