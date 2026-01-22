import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_template_scalar_with_set(self):
    m = self.m
    t = IndexTemplate(m.I)
    e = m.s[t]
    self.assertIs(type(e), EXPR.NPV_Structural_GetItemExpression)
    self.assertEqual(e.args, (m.s, t))
    self.assertFalse(e.is_constant())
    self.assertTrue(e.is_fixed())
    ee = e.polynomial_degree()
    self.assertIs(type(ee), EXPR.CallExpression)
    t.set_value(1)
    with self.assertRaisesRegex(AttributeError, "'_InsertionOrderSetData' object has no attribute 'polynomial_degree'"):
        e.polynomial_degree()
    self.assertEqual(str(e), 's[{I}]')
    t.set_value(5)
    v = e()
    self.assertIs(v, m.s[5])
    self.assertIs(resolve_template(e), m.s[5])
    t.set_value()