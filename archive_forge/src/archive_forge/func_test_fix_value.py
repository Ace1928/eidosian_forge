from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.core import ConcreteModel, Var, Constraint
from pyomo.gdp import Disjunction, Disjunct
from pyomo.gdp.disjunct import AutoLinkedBooleanVar, AutoLinkedBinaryVar
def test_fix_value(self):
    m = ConcreteModel()
    m.iv = AutoLinkedBooleanVar()
    m.biv = AutoLinkedBinaryVar(m.iv)
    m.iv.associate_binary_var(m.biv)
    m.iv.fix()
    self.assertTrue(m.iv.is_fixed())
    self.assertTrue(m.biv.is_fixed())
    self.assertIsNone(m.iv.value)
    self.assertIsNone(m.biv.value)
    m.iv.fix(True)
    self.assertEqual(m.iv.value, True)
    self.assertEqual(m.biv.value, 1)
    m.iv.fix(False)
    self.assertEqual(m.iv.value, False)
    self.assertEqual(m.biv.value, 0)
    m.iv.fix(None)
    self.assertEqual(m.iv.value, None)
    self.assertEqual(m.biv.value, None)
    m.biv.fix(1)
    self.assertEqual(m.iv.value, True)
    self.assertEqual(m.biv.value, 1)
    with LoggingIntercept() as LOG:
        m.biv.fix(0.5)
    self.assertEqual(LOG.getvalue().strip(), "Setting Var 'biv' to a value `0.5` (float) not in domain Binary.")
    self.assertEqual(m.iv.value, None)
    self.assertEqual(m.biv.value, 0.5)
    with LoggingIntercept() as LOG:
        m.biv.fix(0.55, True)
    self.assertEqual(LOG.getvalue().strip(), '')
    self.assertEqual(m.iv.value, None)
    self.assertEqual(m.biv.value, 0.55)
    m.biv.fix(0)
    self.assertEqual(m.iv.value, False)
    self.assertEqual(m.biv.value, 0)
    eps = AutoLinkedBinaryVar.INTEGER_TOLERANCE / 10
    with LoggingIntercept() as LOG:
        m.biv.fix(1 - eps)
    self.assertEqual(LOG.getvalue().strip(), "Setting Var 'biv' to a value `%s` (float) not in domain Binary." % (1 - eps))
    self.assertEqual(m.iv.value, True)
    self.assertEqual(m.biv.value, 1 - eps)
    with LoggingIntercept() as LOG:
        m.biv.fix(eps, True)
    self.assertEqual(LOG.getvalue().strip(), '')
    self.assertEqual(m.iv.value, False)
    self.assertEqual(m.biv.value, eps)
    m.iv.fix(True)
    self.assertEqual(m.iv.value, True)
    self.assertEqual(m.biv.value, 1)
    m.iv.fix(False)
    self.assertEqual(m.iv.value, False)
    self.assertEqual(m.biv.value, 0)