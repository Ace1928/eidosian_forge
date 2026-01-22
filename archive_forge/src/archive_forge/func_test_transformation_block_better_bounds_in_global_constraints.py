import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.logical_expr import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error, check_model_algebraic
from pyomo.gdp.plugins.partition_disjuncts import (
from pyomo.core import Block, value
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.common_tests as ct
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.opt import check_available_solvers
@unittest.skipIf('gurobi_direct' not in solvers, 'Gurobi direct solver not available')
def test_transformation_block_better_bounds_in_global_constraints(self):
    m = models.makeBetweenStepsPaperExample()
    m.c1 = Constraint(expr=m.x[1] ** 2 + m.x[2] ** 2 <= 32)
    m.c2 = Constraint(expr=m.x[3] ** 2 + m.x[4] ** 2 <= 32)
    m.c3 = Constraint(expr=(3 - m.x[1]) ** 2 + (3 - m.x[2]) ** 2 <= 32)
    m.c4 = Constraint(expr=(3 - m.x[3]) ** 2 + (3 - m.x[4]) ** 2 <= 32)
    opt = SolverFactory('gurobi_direct')
    opt.options['NonConvex'] = 2
    opt.options['FeasibilityTol'] = 1e-08
    TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_solver=opt, compute_bounds_method=compute_optimal_bounds)
    self.check_transformation_block(m, 0, 32, 0, 32, -18, 14, -18, 14, partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]])