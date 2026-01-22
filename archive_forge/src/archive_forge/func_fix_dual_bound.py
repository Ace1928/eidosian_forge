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
def fix_dual_bound(self, last_iter_cuts):
    """Fix the dual bound when no-good cuts or tabu list is activated.

        Parameters
        ----------
        last_iter_cuts : bool
            Whether the cuts in the last iteration have been added.
        """
    config = self.config
    if config.single_tree:
        config.logger.info('Fix the bound to the value of one iteration before optimal solution is found.')
        try:
            self.dual_bound = self.stored_bound[self.primal_bound]
        except KeyError as e:
            config.logger.error(e, exc_info=True)
            config.logger.error('No stored bound found. Bound fix failed.')
    else:
        config.logger.info('Solve the main problem without the last no_good cut to fix the bound.zero_tolerance is set to 1E-4')
        config.zero_tolerance = 0.0001
        if not last_iter_cuts:
            fixed_nlp, fixed_nlp_result = self.solve_subproblem()
            self.handle_nlp_subproblem_tc(fixed_nlp, fixed_nlp_result)
        MindtPy = self.mip.MindtPy_utils
        self.deactivate_no_good_cuts_when_fixing_bound(MindtPy.cuts.no_good_cuts)
        if config.add_regularization is not None and MindtPy.component('mip_obj') is None:
            MindtPy.objective_list[-1].activate()
        if isinstance(self.mip_opt, PersistentSolver):
            self.mip_opt.set_instance(self.mip, symbolic_solver_labels=True)
        mip_args = dict(config.mip_solver_args)
        update_solver_timelimit(self.mip_opt, config.mip_solver, self.timing, config)
        main_mip_results = self.mip_opt.solve(self.mip, tee=config.mip_solver_tee, load_solutions=self.load_solutions, **mip_args)
        if len(main_mip_results.solution) > 0:
            self.mip.solutions.load_from(main_mip_results)
        if main_mip_results.solver.termination_condition is tc.infeasible:
            config.logger.info('Bound fix failed. The bound fix problem is infeasible')
        else:
            self.update_suboptimal_dual_bound(main_mip_results)
            config.logger.info('Fixed bound values: Primal Bound: {}  Dual Bound: {}'.format(self.primal_bound, self.dual_bound))
        if abs(self.primal_bound - self.dual_bound) <= config.absolute_bound_tolerance:
            self.results.solver.termination_condition = tc.optimal