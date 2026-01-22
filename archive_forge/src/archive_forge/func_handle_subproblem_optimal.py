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
def handle_subproblem_optimal(self, fixed_nlp, cb_opt=None, fp=False):
    """This function copies the result of the NLP solver function ('solve_subproblem') to the working model, updates
        the bounds, adds OA and no-good cuts, and then stores the new solution if it is the new best solution. This
        function handles the result of the latest iteration of solving the NLP subproblem given an optimal solution.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        cb_opt : SolverFactory, optional
            The gurobi_persistent solver, by default None.
        fp : bool, optional
            Whether it is in the loop of feasibility pump, by default False.
        """
    config = self.config
    copy_var_list_values(fixed_nlp.MindtPy_utils.variable_list, self.working_model.MindtPy_utils.variable_list, config)
    if config.calculate_dual_at_solution:
        for c in fixed_nlp.tmp_duals:
            if fixed_nlp.dual.get(c, None) is None:
                fixed_nlp.dual[c] = fixed_nlp.tmp_duals[c]
            elif self.config.nlp_solver == 'cyipopt' and self.objective_sense == minimize:
                fixed_nlp.dual[c] = -fixed_nlp.dual[c]
        dual_values = list((fixed_nlp.dual[c] for c in fixed_nlp.MindtPy_utils.constraint_list))
    else:
        dual_values = None
    main_objective = fixed_nlp.MindtPy_utils.objective_list[-1]
    self.update_primal_bound(value(main_objective.expr))
    if self.primal_bound_improved:
        self.best_solution_found = fixed_nlp.clone()
        self.best_solution_found_time = get_main_elapsed_time(self.timing)
    copy_var_list_values(fixed_nlp.MindtPy_utils.variable_list, self.mip.MindtPy_utils.variable_list, config)
    self.add_cuts(dual_values=dual_values, linearize_active=True, linearize_violated=True, cb_opt=cb_opt, nlp=self.fixed_nlp)
    var_values = list((v.value for v in fixed_nlp.MindtPy_utils.variable_list))
    if config.add_no_good_cuts:
        add_no_good_cuts(self.mip, var_values, config, self.timing, self.mip_iter, cb_opt)
    config.call_after_subproblem_feasible(fixed_nlp)
    config.logger.info(self.fixed_nlp_log_formatter.format('*' if self.primal_bound_improved else ' ', self.nlp_iter if not fp else self.fp_iter, 'Fixed NLP', value(main_objective.expr), self.primal_bound, self.dual_bound, self.rel_gap, get_main_elapsed_time(self.timing)))