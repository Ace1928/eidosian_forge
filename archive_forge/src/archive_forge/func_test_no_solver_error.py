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
def test_no_solver_error(self):
    m = models.makeBetweenStepsPaperExample()
    with self.assertRaisesRegex(GDP_Error, "No solver was specified to optimize the subproblems for computing expression bounds! Please specify a configured solver in the 'compute_bounds_solver' argument if using 'compute_optimal_bounds.'"):
        TransformationFactory('gdp.partition_disjuncts').apply_to(m, variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_method=compute_optimal_bounds)