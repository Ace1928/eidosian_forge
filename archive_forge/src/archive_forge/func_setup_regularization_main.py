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
def setup_regularization_main(self):
    """Set up main regularization problem for ROA method."""
    config = self.config
    MindtPy = self.mip.MindtPy_utils
    for c in MindtPy.constraint_list:
        if c.body.polynomial_degree() not in self.mip_constraint_polynomial_degree:
            c.deactivate()
    MindtPy.cuts.activate()
    sign_adjust = 1 if self.objective_sense == minimize else -1
    MindtPy.del_component('mip_obj')
    if config.single_tree:
        MindtPy.del_component('roa_proj_mip_obj')
        MindtPy.cuts.del_component('obj_reg_estimate')
    if config.add_regularization is not None and config.add_no_good_cuts:
        MindtPy.cuts.no_good_cuts.activate()
    if MindtPy.objective_list[0].expr.polynomial_degree() in self.mip_objective_polynomial_degree:
        MindtPy.objective_constr.activate()
    if config.add_regularization == 'level_L1':
        MindtPy.roa_proj_mip_obj = generate_norm1_objective_function(self.mip, self.best_solution_found, discrete_only=False)
    elif config.add_regularization == 'level_L2':
        MindtPy.roa_proj_mip_obj = generate_norm2sq_objective_function(self.mip, self.best_solution_found, discrete_only=False)
    elif config.add_regularization == 'level_L_infinity':
        MindtPy.roa_proj_mip_obj = generate_norm_inf_objective_function(self.mip, self.best_solution_found, discrete_only=False)
    elif config.add_regularization in {'grad_lag', 'hess_lag', 'hess_only_lag', 'sqp_lag'}:
        MindtPy.roa_proj_mip_obj = generate_lag_objective_function(self.mip, self.best_solution_found, config, self.timing, discrete_only=False)
    if self.objective_sense == minimize:
        MindtPy.cuts.obj_reg_estimate = Constraint(expr=sum(MindtPy.objective_value[:]) <= (1 - config.level_coef) * self.primal_bound + config.level_coef * self.dual_bound)
    else:
        MindtPy.cuts.obj_reg_estimate = Constraint(expr=sum(MindtPy.objective_value[:]) >= (1 - config.level_coef) * self.primal_bound + config.level_coef * self.dual_bound)