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
def set_up_solve_data(self, model):
    """Set up the solve data.

        Parameters
        ----------
        model : Pyomo model
            The original model to be solved in MindtPy.
        """
    config = self.config
    obj = next(model.component_data_objects(ctype=Objective, active=True))
    if obj.expr.polynomial_degree() == 0:
        config.logger.info('The model has a constant objecitive function. use_dual_bound is set to False.')
        config.use_dual_bound = False
    if config.use_fbbt:
        fbbt(model)
        config.logger.info('Use the fbbt to tighten the bounds of variables')
    self.original_model = model
    self.working_model = model.clone()
    if obj.sense == minimize:
        self.primal_bound = float('inf')
        self.dual_bound = float('-inf')
    else:
        self.primal_bound = float('-inf')
        self.dual_bound = float('inf')
    self.primal_bound_progress = [self.primal_bound]
    self.dual_bound_progress = [self.dual_bound]
    if config.nlp_solver in {'ipopt', 'cyipopt'}:
        if not hasattr(self.working_model, 'ipopt_zL_out'):
            self.working_model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        if not hasattr(self.working_model, 'ipopt_zU_out'):
            self.working_model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    if config.quadratic_strategy == 0:
        self.mip_objective_polynomial_degree = {0, 1}
        self.mip_constraint_polynomial_degree = {0, 1}
    elif config.quadratic_strategy == 1:
        self.mip_objective_polynomial_degree = {0, 1, 2}
        self.mip_constraint_polynomial_degree = {0, 1}
    elif config.quadratic_strategy == 2:
        self.mip_objective_polynomial_degree = {0, 1, 2}
        self.mip_constraint_polynomial_degree = {0, 1, 2}