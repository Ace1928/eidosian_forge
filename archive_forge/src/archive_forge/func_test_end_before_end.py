import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_end_before_end(self):
    m = self.get_model()
    m.c = LogicalConstraint(expr=m.i.end_time.before(m.i2[1].end_time, delay=6))
    visitor = self.get_visitor()
    expr = visitor.walk_expression((m.c.body, m.c, 0))
    self.assertIn(id(m.i), visitor.var_map)
    self.assertIn(id(m.i2[1]), visitor.var_map)
    i = visitor.var_map[id(m.i)]
    i21 = visitor.var_map[id(m.i2[1])]
    self.assertTrue(expr[1].equals(cp.end_before_end(i, i21, 6)))