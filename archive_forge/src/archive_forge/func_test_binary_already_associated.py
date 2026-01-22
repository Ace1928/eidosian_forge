from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def test_binary_already_associated(self):
    m = self.make_model()
    m.mine = Var(domain=Binary)
    m.a.associate_binary_var(m.mine)
    e = m.a.land(m.b)
    visitor = LogicalToDisjunctiveVisitor()
    m.cons = visitor.constraints
    m.z = visitor.z_vars
    visitor.walk_expression(e)
    self.assertEqual(len(m.z), 2)
    self.assertIs(m.b.get_associated_binary(), m.z[1])
    self.assertEqual(len(m.cons), 4)
    assertExpressionsEqual(self, m.cons[1].expr, m.z[2] <= m.mine)
    assertExpressionsEqual(self, m.cons[2].expr, m.z[2] <= m.z[1])
    assertExpressionsEqual(self, m.cons[3].expr, 1 - m.z[2] <= 2 - (m.mine + m.z[1]))
    assertExpressionsEqual(self, m.cons[4].expr, m.z[2] >= 1)