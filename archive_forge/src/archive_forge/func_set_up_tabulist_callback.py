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
def set_up_tabulist_callback(self):
    """Sets up the tabulist using IncumbentCallback.
        Currently only support CPLEX.
        """
    tabulist = self.mip_opt._solver_model.register_callback(tabu_list.IncumbentCallback_cplex)
    tabulist.opt = self.mip_opt
    tabulist.config = self.config
    tabulist.mindtpy_solver = self
    self.mip_opt.options['preprocessing_reduce'] = 1
    self.mip_opt._solver_model.set_warning_stream(None)
    self.mip_opt._solver_model.set_log_stream(None)
    self.mip_opt._solver_model.set_error_stream(None)