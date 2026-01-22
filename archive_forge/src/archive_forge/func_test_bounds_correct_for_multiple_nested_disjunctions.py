from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_bounds_correct_for_multiple_nested_disjunctions(self):
    m = ConcreteModel()
    m.c = Var(bounds=(3, 9))
    m.x = Var(bounds=(0, 10))
    m.y = Var(bounds=(-10, 2))
    m.d1 = Disjunct()
    m.d1.cons = Constraint(expr=m.c == 4)
    m.d1.disjunction = Disjunction(expr=[[m.x + m.y >= 8], [m.x + m.y <= 3]])
    m.d1.disjunction2 = Disjunction(expr=[[m.x + 2 * m.y <= 4], [m.y + 2 * m.x >= 7]])
    m.d2 = Disjunct()
    m.d2.cons = Constraint(expr=m.c == 5)
    m.d2.disjunction = Disjunction(expr=[[m.x + m.y >= 10], [m.x + m.y <= 0]])
    m.d2.disjunction2 = Disjunction(expr=[[m.x + 3 * m.y <= 2], [m.y + 2 * m.x >= 9]])
    m.disjunction = Disjunction(expr=[m.d1, m.d2])
    m.obj = Objective(expr=m.c)
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    cons = bt.get_transformed_constraints(m.x, m.disjunction)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.y, m.disjunction)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.c, m.disjunction)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    assertExpressionsEqual(self, lb.expr, 4.0 * m.d1.binary_indicator_var + 5.0 * m.d2.binary_indicator_var <= m.c)
    ub = cons[1]
    assertExpressionsEqual(self, ub.expr, 4.0 * m.d1.binary_indicator_var + 5.0 * m.d2.binary_indicator_var >= m.c)
    cons = bt.get_transformed_constraints(m.x, m.d1.disjunction)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.y, m.d1.disjunction)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.c, m.d1.disjunction)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.x, m.d1.disjunction2)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.y, m.d1.disjunction2)
    self.assertEqual(len(cons), 0)
    cons = bt.get_transformed_constraints(m.c, m.d1.disjunction2)
    self.assertEqual(len(cons), 0)