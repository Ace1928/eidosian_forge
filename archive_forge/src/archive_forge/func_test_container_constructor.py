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
def test_container_constructor(self):
    um0 = PyomoUnitsContainer(None)
    self.assertIsNone(um0.pint_registry)
    self.assertIsNone(um0._pint_dimensionless)
    um1 = PyomoUnitsContainer()
    self.assertIsNotNone(um1.pint_registry)
    self.assertIsNotNone(um1._pint_dimensionless)
    with self.assertRaisesRegex(ValueError, 'Cannot operate with Unit and Unit of different registries'):
        self.assertEqual(um1._pint_dimensionless, units._pint_dimensionless)
    self.assertIsNot(um1.pint_registry, units.pint_registry)
    um2 = PyomoUnitsContainer(pint_module.UnitRegistry())
    self.assertIsNotNone(um2.pint_registry)
    self.assertIsNotNone(um2._pint_dimensionless)
    with self.assertRaisesRegex(ValueError, 'Cannot operate with Unit and Unit of different registries'):
        self.assertEqual(um2._pint_dimensionless, units._pint_dimensionless)
    self.assertIsNot(um2.pint_registry, units.pint_registry)
    self.assertIsNot(um2.pint_registry, um1.pint_registry)
    um3 = PyomoUnitsContainer(units.pint_registry)
    self.assertIs(um3.pint_registry, units.pint_registry)
    self.assertEqual(um3._pint_dimensionless, units._pint_dimensionless)