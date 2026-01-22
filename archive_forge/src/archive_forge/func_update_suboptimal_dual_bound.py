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
def update_suboptimal_dual_bound(self, results):
    """If the relaxed problem is not solved to optimality, the dual bound is updated
        according to the dual bound of relaxed problem.

        Parameters
        ----------
        results : SolverResults
            Results from solving the relaxed problem.
            The dual bound of the relaxed problem can only be obtained from the result object.
        """
    if self.objective_sense == minimize:
        bound_value = results.problem.lower_bound
    else:
        bound_value = results.problem.upper_bound
    self.update_dual_bound(bound_value)