import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_using_precedence_expr_as_boolean_expr_positive_delay(self):
    m = self.get_model()
    e = m.b.implies(m.i2[2].start_time.before(m.i2[1].start_time, delay=4))
    visitor = self.get_visitor()
    expr = visitor.walk_expression((e, e, 0))
    self.assertIn(id(m.b), visitor.var_map)
    self.assertIn(id(m.i2[1]), visitor.var_map)
    self.assertIn(id(m.i2[2]), visitor.var_map)
    b = visitor.var_map[id(m.b)]
    i21 = visitor.var_map[id(m.i2[1])]
    i22 = visitor.var_map[id(m.i2[2])]
    self.assertTrue(expr[1].equals(cp.if_then(b, cp.start_of(i22) + 4 <= cp.start_of(i21))))