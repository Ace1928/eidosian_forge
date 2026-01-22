from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def test_boolean_fixed_none(self):
    m = self.make_model()
    e = m.a & m.b
    m.a.fix(None)
    visitor = LogicalToDisjunctiveVisitor()
    m.cons = visitor.constraints
    m.z = visitor.z_vars
    m.disjuncts = visitor.disjuncts
    m.disjunctions = visitor.disjunctions
    visitor.walk_expression(e)
    self.assertEqual(len(m.z), 3)
    self.assertEqual(len(m.cons), 4)
    self.assertIs(m.a.get_associated_binary(), m.z[1])
    self.assertTrue(m.z[1].fixed)
    self.assertIsNone(m.z[1].value)
    self.assertIs(m.b.get_associated_binary(), m.z[2])
    assertExpressionsEqual(self, m.cons[1].expr, m.z[1] >= m.z[3])
    assertExpressionsEqual(self, m.cons[2].expr, m.z[2] >= m.z[3])
    assertExpressionsEqual(self, m.cons[3].expr, 1 - m.z[3] <= 2 - (m.z[1] + m.z[2]))
    assertExpressionsEqual(self, m.cons[4].expr, m.z[3] >= 1)