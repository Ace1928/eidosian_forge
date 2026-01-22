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
def reached_time_limit(self):
    if get_main_elapsed_time(self.timing) >= self.config.time_limit:
        self.config.logger.info('MindtPy unable to converge bounds before time limit of {} seconds. Elapsed: {} seconds'.format(self.config.time_limit, get_main_elapsed_time(self.timing)))
        self.config.logger.info('Final bound values: Primal Bound: {}  Dual Bound: {}'.format(self.primal_bound, self.dual_bound))
        self.results.solver.termination_condition = tc.maxTimeLimit
        return True
    else:
        return False