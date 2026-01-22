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
def get_primal_integral(self):
    """Calculate the primal integral.
        Ref: The confined primal integral. [http://www.optimization-online.org/DB_FILE/2020/07/7910.pdf]

        Returns
        -------
        float
            The primal integral.
        """
    primal_integral = 0
    primal_bound_progress = self.primal_bound_progress.copy()
    for primal_bound in primal_bound_progress:
        if primal_bound != primal_bound_progress[0]:
            break
    for i in range(len(primal_bound_progress)):
        if primal_bound_progress[i] == self.primal_bound_progress[0]:
            primal_bound_progress[i] = primal_bound * (1 + self.config.initial_bound_coef * self.objective_sense * math.copysign(1, primal_bound))
        else:
            break
    for i in range(len(primal_bound_progress)):
        if i == 0:
            primal_integral += abs(primal_bound_progress[i] - self.primal_bound) * self.primal_bound_progress_time[i]
        else:
            primal_integral += abs(primal_bound_progress[i] - self.primal_bound) * (self.primal_bound_progress_time[i] - self.primal_bound_progress_time[i - 1])
    self.config.logger.info(' {:<25}:   {:>7.4f} '.format('Primal integral', primal_integral))
    return primal_integral