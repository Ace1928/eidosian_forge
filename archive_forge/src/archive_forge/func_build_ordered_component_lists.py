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
def build_ordered_component_lists(self, model):
    """Define lists used for future data transfer.

        Also attaches ordered lists of the variables, constraints to the model so that they can be used for mapping back and
        forth.

        """
    util_block = getattr(model, self.util_block_name)
    var_set = ComponentSet()
    util_block.constraint_list = list(model.component_data_objects(ctype=Constraint, active=True, descend_into=Block))
    if egb_available:
        util_block.grey_box_list = list(model.component_data_objects(ctype=egb.ExternalGreyBoxBlock, active=True, descend_into=Block))
    else:
        util_block.grey_box_list = []
    util_block.linear_constraint_list = list((c for c in util_block.constraint_list if c.body.polynomial_degree() in self.mip_constraint_polynomial_degree))
    util_block.nonlinear_constraint_list = list((c for c in util_block.constraint_list if c.body.polynomial_degree() not in self.mip_constraint_polynomial_degree))
    util_block.objective_list = list(model.component_data_objects(ctype=Objective, active=True, descend_into=Block))
    for constr in getattr(util_block, 'constraint_list'):
        for v in EXPR.identify_variables(constr.body, include_fixed=False):
            var_set.add(v)
    for obj in model.component_data_objects(ctype=Objective, active=True):
        for v in EXPR.identify_variables(obj.expr, include_fixed=False):
            var_set.add(v)
    if egb_available:
        util_block.variable_list = list((v for v in model.component_data_objects(ctype=Var, descend_into=(Block, egb.ExternalGreyBoxBlock)) if v in var_set))
    else:
        util_block.variable_list = list((v for v in model.component_data_objects(ctype=Var, descend_into=Block) if v in var_set))
    util_block.discrete_variable_list = list((v for v in util_block.variable_list if v in var_set and v.is_integer()))
    util_block.continuous_variable_list = list((v for v in util_block.variable_list if v in var_set and v.is_continuous()))