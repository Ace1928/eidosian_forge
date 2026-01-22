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
def handle_nlp_subproblem_tc(self, fixed_nlp, result, cb_opt=None):
    """This function handles different terminaton conditions of the fixed-NLP subproblem.

        Parameters
        ----------
        fixed_nlp : Pyomo model
            Integer-variable-fixed NLP model.
        result : SolverResults
            Results from solving the NLP subproblem.
        cb_opt : SolverFactory, optional
            The gurobi_persistent solver, by default None.
        """
    if result.solver.termination_condition in {tc.optimal, tc.locallyOptimal, tc.feasible}:
        self.handle_subproblem_optimal(fixed_nlp, cb_opt)
    elif result.solver.termination_condition in {tc.infeasible, tc.noSolution}:
        self.handle_subproblem_infeasible(fixed_nlp, cb_opt)
    elif result.solver.termination_condition is tc.maxTimeLimit:
        self.config.logger.info('NLP subproblem failed to converge within the time limit.')
        self.results.solver.termination_condition = tc.maxTimeLimit
        self.should_terminate = True
    elif result.solver.termination_condition is tc.maxEvaluations:
        self.config.logger.info('NLP subproblem failed due to maxEvaluations.')
        self.results.solver.termination_condition = tc.maxEvaluations
        self.should_terminate = True
    else:
        self.handle_subproblem_other_termination(fixed_nlp, result.solver.termination_condition, cb_opt)