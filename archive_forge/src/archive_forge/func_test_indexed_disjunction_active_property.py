from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core import ConcreteModel, Var, Constraint
from pyomo.gdp import Disjunction, Disjunct
from pyomo.gdp.disjunct import AutoLinkedBooleanVar, AutoLinkedBinaryVar
def test_indexed_disjunction_active_property(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 12))

    @m.Disjunction([0, 1, 2])
    def disjunction(m, i):
        return [m.x == i * 5, m.x == i * 5 + 1]
    self.assertTrue(m.disjunction.active)
    m.disjunction[2].deactivate()
    self.assertTrue(m.disjunction.active)
    m.disjunction[0].deactivate()
    m.disjunction[1].deactivate()
    self.assertFalse(m.disjunction.active)
    m.disjunction.activate()
    self.assertTrue(m.disjunction.active)
    m.disjunction.deactivate()
    self.assertFalse(m.disjunction.active)
    for i in range(3):
        self.assertFalse(m.disjunction[i].active)