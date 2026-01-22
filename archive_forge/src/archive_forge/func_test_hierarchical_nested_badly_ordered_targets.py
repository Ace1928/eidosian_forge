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
def test_hierarchical_nested_badly_ordered_targets(self):
    m = models.makeHierarchicalNested_DeclOrderMatchesInstantiationOrder()
    TransformationFactory('gdp.partition_disjuncts').apply_to(m, targets=[m.disjunction_block, m.disjunct_block.disj2], variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]], compute_bounds_method=compute_fbbt_bounds)
    self.check_hierarchical_nested_model(m)