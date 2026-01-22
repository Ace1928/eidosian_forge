import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_simple_sum_rule(self):
    m = ConcreteModel()
    m.I = RangeSet(3)
    m.J = RangeSet(3)
    m.x = Var(m.I, m.J)

    @m.Constraint(m.I)
    def c(m, i):
        return sum((m.x[i, j] for j in m.J)) <= 0
    template, indices = templatize_constraint(m.c)
    self.assertEqual(len(indices), 1)
    self.assertIs(indices[0]._set, m.I)
    self.assertEqual(template.to_string(verbose=True), 'templatesum(getitem(x, _1, _2), iter(_2, J))  <=  0')
    self.assertEqual(str(template), 'SUM(x[_1,_2] for _2 in J)  <=  0')
    indices[0].set_value(2)
    self.assertEqual(str(resolve_template(template)), 'x[2,1] + x[2,2] + x[2,3]  <=  0')