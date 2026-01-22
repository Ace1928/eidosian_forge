import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_simple_rule_nonfinite_set(self):
    m = ConcreteModel()
    m.x = Var(Integers, dense=False)

    @m.Constraint(Integers)
    def c(m, i):
        return m.x[i] <= 0
    template, indices = templatize_constraint(m.c)
    self.assertEqual(len(indices), 1)
    self.assertIs(indices[0]._set, Integers)
    self.assertEqual(str(template), 'x[_1]  <=  0')
    indices[0].set_value(2)
    self.assertEqual(str(resolve_template(template)), 'x[2]  <=  0')