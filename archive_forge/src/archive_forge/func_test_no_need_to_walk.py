from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def test_no_need_to_walk(self):
    m = self.make_model()
    e = m.a
    visitor = LogicalToDisjunctiveVisitor()
    m.cons = visitor.constraints
    m.z = visitor.z_vars
    visitor.walk_expression(e)
    self.assertEqual(len(m.z), 1)
    self.assertIs(m.a.get_associated_binary(), m.z[1])
    self.assertEqual(len(m.cons), 1)
    assertExpressionsEqual(self, m.cons[1].expr, m.z[1] >= 1)