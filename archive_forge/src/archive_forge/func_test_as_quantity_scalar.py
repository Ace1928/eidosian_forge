import math
import pickle
from pyomo.common.errors import PyomoException
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.util.check_units import assert_units_consistent, check_units_equivalent
from pyomo.core.expr import inequality
from pyomo.core.expr.numvalue import NumericConstant
import pyomo.core.expr as EXPR
from pyomo.core.base.units_container import (
from io import StringIO
def test_as_quantity_scalar(self):
    _pint = units._pint_registry
    Quantity = _pint.Quantity
    m = ConcreteModel()
    m.x = Var(initialize=1)
    m.y = Var(initialize=2, units=units.g)
    m.p = Param(initialize=3)
    m.q = Param(initialize=4, units=1 / units.s)
    m.b = BooleanVar(initialize=True)
    q = as_quantity(0)
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 0 * _pint.dimensionless)
    q = as_quantity(None)
    self.assertIs(q.__class__, None.__class__)
    self.assertEqual(q, None)
    q = as_quantity(str('aaa'))
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 'aaa' * _pint.dimensionless)
    q = as_quantity(True)
    self.assertIs(q.__class__, bool)
    self.assertEqual(q, True)
    q = as_quantity(units.kg)
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 1 * _pint.kg)
    q = as_quantity(NumericConstant(5))
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 5 * _pint.dimensionless)
    q = as_quantity(m.x)
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 1 * _pint.dimensionless)
    q = as_quantity(m.y)
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 2 * _pint.g)
    q = as_quantity(m.p)
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 3 * _pint.dimensionless)
    q = as_quantity(m.q)
    self.assertIs(q.__class__, Quantity)
    self.assertEqual(q, 4 / _pint.s)
    q = as_quantity(m.b)
    self.assertIs(q.__class__, bool)
    self.assertEqual(q, True)

    class UnknownPyomoType(object):

        def is_expression_type(self, expression_system=None):
            return False

        def is_numeric_type(self):
            return False

        def is_logical_type(self):
            return False
    other = UnknownPyomoType()
    q = as_quantity(other)
    self.assertIs(q.__class__, UnknownPyomoType)
    self.assertIs(q, other)