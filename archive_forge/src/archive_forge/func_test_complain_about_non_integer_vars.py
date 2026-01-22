import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_complain_about_non_integer_vars(self):
    m = self.get_model()
    m.c = LogicalConstraint(expr=m.i.is_present.implies(m.a[1] == 5))
    visitor = self.get_visitor()
    with self.assertRaisesRegex(ValueError, "The LogicalToDoCplex writer can only support integer- or Boolean-valued variables. Cannot write Var 'a\\[1\\]' with domain 'Reals'"):
        expr = visitor.walk_expression((m.c.expr, m.c, 0))