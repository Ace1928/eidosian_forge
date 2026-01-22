import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_interval_var_fixed_presences_correct(self):
    m = self.get_model()
    m.silly = LogicalConstraint(expr=m.i.is_present)
    visitor = self.get_visitor()
    expr = visitor.walk_expression((m.silly.expr, m.silly, 0))
    self.assertIn(id(m.i), visitor.var_map)
    i = visitor.var_map[id(m.i)]
    self.assertTrue(i.is_optional())
    m.i.is_present.fix(False)
    m.c = LogicalConstraint(expr=m.i.is_present.lor(m.i2[1].start_time == 2))
    visitor = self.get_visitor()
    expr = visitor.walk_expression((m.c.body, m.c, 0))
    self.assertIn(id(m.i2[1]), visitor.var_map)
    i21 = visitor.var_map[id(m.i2[1])]
    self.assertIn(id(m.i), visitor.var_map)
    i = visitor.var_map[id(m.i)]
    self.assertTrue(i.is_absent())
    self.assertTrue(i21.is_present())