import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import IntegerSet
from pyomo.core.expr.numeric_expr import (
from pyomo.core.staleflag import StaleFlagManager
from pyomo.environ import (
from pyomo.core.base.units_container import units, pint_available, UnitsError
def test_ub(self):
    m = ConcreteModel()
    m.x = Var()
    self.assertEqual(m.x.ub, None)
    m.x.domain = NonPositiveReals
    self.assertEqual(m.x.ub, 0)
    m.x.ub = float('-inf')
    with self.assertRaisesRegex(ValueError, 'invalid non-finite upper bound \\(-inf\\)'):
        m.x.ub
    m.x.ub = float('nan')
    with self.assertRaisesRegex(ValueError, 'invalid non-finite upper bound \\(nan\\)'):
        m.x.ub