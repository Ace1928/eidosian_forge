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
def handle_regularization_main_tc(self, main_mip, main_mip_results):
    """Handles the result of the regularization main problem.

        Parameters
        ----------
        main_mip : Pyomo model
            The MIP main problem.
        main_mip_results : SolverResults
            Results from solving the regularization main subproblem.

        Raises
        ------
        ValueError
            MindtPy unable to handle the regularization problem termination condition.
        """
    if main_mip_results is None:
        self.config.logger.info('Failed to solve the regularization problem.The solution of the OA main problem will be adopted.')
    elif main_mip_results.solver.termination_condition in {tc.optimal, tc.feasible}:
        self.handle_main_optimal(main_mip, update_bound=False)
    elif main_mip_results.solver.termination_condition is tc.maxTimeLimit:
        self.config.logger.info('Regularization problem failed to converge within the time limit.')
        self.results.solver.termination_condition = tc.maxTimeLimit
    elif main_mip_results.solver.termination_condition is tc.infeasible:
        self.config.logger.info('Regularization problem infeasible.')
    elif main_mip_results.solver.termination_condition is tc.unbounded:
        self.config.logger.info('Regularization problem unbounded.Sometimes solving MIQCP in CPLEX, unbounded means infeasible.')
    elif main_mip_results.solver.termination_condition is tc.infeasibleOrUnbounded:
        self.config.logger.info('Regularization problem is infeasible or unbounded.It might happen when using CPLEX to solve MIQP.')
    elif main_mip_results.solver.termination_condition is tc.unknown:
        self.config.logger.info('Termination condition of the regularization problem is unknown.')
        if main_mip_results.problem.lower_bound != float('-inf'):
            self.config.logger.info('Solution limit has been reached.')
            self.handle_main_optimal(main_mip, update_bound=False)
        else:
            self.config.logger.info('No solution obtained from the regularization subproblem.Please set mip_solver_tee to True for more information.The solution of the OA main problem will be adopted.')
    else:
        raise ValueError('MindtPy unable to handle regularization problem termination condition of %s. Solver message: %s' % (main_mip_results.solver.termination_condition, main_mip_results.solver.message))