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
def set_appsi_solver_update_config(self):
    """Set update config for APPSI solvers."""
    config = self.config
    if config.mip_solver in {'appsi_cplex', 'appsi_gurobi', 'appsi_highs'}:
        self.mip_opt.update_config.check_for_new_or_removed_constraints = True
        self.mip_opt.update_config.check_for_new_or_removed_vars = True
        self.mip_opt.update_config.check_for_new_or_removed_params = False
        self.mip_opt.update_config.check_for_new_objective = True
        self.mip_opt.update_config.update_constraints = True
        self.mip_opt.update_config.update_vars = True
        self.mip_opt.update_config.update_params = False
        self.mip_opt.update_config.update_named_expressions = False
        self.mip_opt.update_config.update_objective = False
        self.mip_opt.update_config.treat_fixed_vars_as_params = True
    if config.nlp_solver == 'appsi_ipopt':
        self.nlp_opt.update_config.check_for_new_or_removed_constraints = False
        self.nlp_opt.update_config.check_for_new_or_removed_vars = False
        self.nlp_opt.update_config.check_for_new_or_removed_params = False
        self.nlp_opt.update_config.check_for_new_objective = False
        self.nlp_opt.update_config.update_constraints = True
        self.nlp_opt.update_config.update_vars = True
        self.nlp_opt.update_config.update_params = False
        self.nlp_opt.update_config.update_named_expressions = False
        self.nlp_opt.update_config.update_objective = False
        self.nlp_opt.update_config.treat_fixed_vars_as_params = False
        self.feasibility_nlp_opt.update_config.check_for_new_or_removed_constraints = False
        self.feasibility_nlp_opt.update_config.check_for_new_or_removed_vars = False
        self.feasibility_nlp_opt.update_config.check_for_new_or_removed_params = False
        self.feasibility_nlp_opt.update_config.check_for_new_objective = False
        self.feasibility_nlp_opt.update_config.update_constraints = False
        self.feasibility_nlp_opt.update_config.update_vars = True
        self.feasibility_nlp_opt.update_config.update_params = False
        self.feasibility_nlp_opt.update_config.update_named_expressions = False
        self.feasibility_nlp_opt.update_config.update_objective = False
        self.feasibility_nlp_opt.update_config.treat_fixed_vars_as_params = False