from pyomo.common.errors import MouseTrap
import pyomo.common.unittest as unittest
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.gdp import Disjunct
from pyomo.environ import (
def test_at_least(self):
    m = self.make_model()
    e = atleast(2, m.a, m.b, m.c)
    visitor = LogicalToDisjunctiveVisitor()
    m.cons = visitor.constraints
    m.z = visitor.z_vars
    m.disjuncts = visitor.disjuncts
    m.disjunctions = visitor.disjunctions
    visitor.walk_expression(e)
    self.assertIs(m.a.get_associated_binary(), m.z[1])
    a = m.z[1]
    self.assertIs(m.b.get_associated_binary(), m.z[2])
    b = m.z[2]
    self.assertIs(m.c.get_associated_binary(), m.z[3])
    c = m.z[3]
    self.assertEqual(len(m.z), 3)
    self.assertEqual(len(m.cons), 1)
    assertExpressionsEqual(self, m.disjuncts[0].constraint.expr, m.z[1] + m.z[2] + m.z[3] >= 2)
    assertExpressionsEqual(self, m.disjuncts[1].constraint.expr, m.z[1] + m.z[2] + m.z[3] <= 1)
    assertExpressionsEqual(self, m.cons[1].expr, m.disjuncts[0].binary_indicator_var >= 1)