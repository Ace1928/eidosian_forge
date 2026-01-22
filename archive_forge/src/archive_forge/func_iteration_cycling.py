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
def iteration_cycling(self):
    config = self.config
    if config.cycling_check or config.use_tabu_list:
        self.curr_int_sol = get_integer_solution(self.mip)
        if config.cycling_check and self.mip_iter >= 1:
            if self.curr_int_sol in set(self.integer_list):
                config.logger.info('Cycling happens after {} main iterations. The same combination is obtained in iteration {} This issue happens when the NLP subproblem violates constraint qualification. Convergence to optimal solution is not guaranteed.'.format(self.mip_iter, self.integer_list.index(self.curr_int_sol) + 1))
                config.logger.info('Final bound values: Primal Bound: {}  Dual Bound: {}'.format(self.primal_bound, self.dual_bound))
                self.results.solver.termination_condition = tc.feasible
                return True
        self.integer_list.append(self.curr_int_sol)
    return False