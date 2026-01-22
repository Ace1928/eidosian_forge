from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_bounds_on_disjuncts_with_block_hierarchies(self):
    m = ConcreteModel()
    m.x = Var()
    m.b = Block()
    m.b.c = Constraint(expr=m.x <= 4)
    m.d = Disjunct([1, 2])
    m.d[1].b = Block()
    m.d[1].b.c = Constraint(expr=m.x <= 5)
    m.d[2].b = Block()
    m.d[2].b.c = Constraint(expr=m.x <= 3)
    m.d[2].c = Constraint(expr=m.x <= 4.1)
    m.disjunction = Disjunction(expr=[m.d[1], m.d[2]])
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    cons = bt.get_transformed_constraints(m.x, m.disjunction)
    self.assertEqual(len(cons), 1)
    ub = cons[0]
    assertExpressionsEqual(self, ub.expr, 4.0 * m.d[1].binary_indicator_var + 3.0 * m.d[2].binary_indicator_var >= m.x)
    self.assertEqual(len(list(m.component_data_objects(Constraint, descend_into=(Block, Disjunct), active=True))), 2)