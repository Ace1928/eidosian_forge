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
def handle_fp_main_tc(self, fp_main_results):
    """Handle the termination condition of the feasibility pump main problem.

        Parameters
        ----------
        fp_main_results : SolverResults
            The results from solving the FP main problem.

        Returns
        -------
        bool
            True if FP loop should terminate, False otherwise.
        """
    if fp_main_results.solver.termination_condition is tc.optimal:
        self.config.logger.info(self.log_formatter.format(self.fp_iter, 'FP-MIP', value(self.mip.MindtPy_utils.fp_mip_obj), self.primal_bound, self.dual_bound, self.rel_gap, get_main_elapsed_time(self.timing)))
        return False
    elif fp_main_results.solver.termination_condition is tc.maxTimeLimit:
        self.config.logger.warning('FP-MIP reaches max TimeLimit')
        self.results.solver.termination_condition = tc.maxTimeLimit
        return True
    elif fp_main_results.solver.termination_condition is tc.infeasible:
        self.config.logger.warning('FP-MIP infeasible')
        no_good_cuts = self.mip.MindtPy_utils.cuts.no_good_cuts
        if no_good_cuts.__len__() > 0:
            no_good_cuts[no_good_cuts.__len__()].deactivate()
        return True
    elif fp_main_results.solver.termination_condition is tc.unbounded:
        self.config.logger.warning('FP-MIP unbounded')
        return True
    elif fp_main_results.solver.termination_condition is tc.other and fp_main_results.solution.status is SolutionStatus.feasible:
        self.config.logger.warning('MILP solver reported feasible solution of FP-MIP, but not guaranteed to be optimal.')
        return False
    else:
        self.config.logger.warning('Unexpected result of FP-MIP')
        return True