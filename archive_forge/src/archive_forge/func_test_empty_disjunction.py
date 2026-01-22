from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core import ConcreteModel, Var, Constraint
from pyomo.gdp import Disjunction, Disjunct
from pyomo.gdp.disjunct import AutoLinkedBooleanVar, AutoLinkedBinaryVar
def test_empty_disjunction(self):
    m = ConcreteModel()
    m.d = Disjunct()
    m.e = Disjunct()
    m.x1 = Disjunction()
    self.assertEqual(len(m.x1), 0)
    m.x1 = [m.d, m.e]
    self.assertEqual(len(m.x1), 1)
    self.assertEqual(m.x1.disjuncts, [m.d, m.e])
    m.x2 = Disjunction([1, 2, 3, 4])
    self.assertEqual(len(m.x2), 0)
    m.x2[2] = [m.d, m.e]
    self.assertEqual(len(m.x2), 1)
    self.assertEqual(m.x2[2].disjuncts, [m.d, m.e])