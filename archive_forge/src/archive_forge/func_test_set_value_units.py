import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
@unittest.skipUnless(pint_available, 'units test requires pint module')
def test_set_value_units(self):
    m = ConcreteModel()
    m.p = Param(units=units.g)
    m.p = 5
    self.assertEqual(value(m.p), 5)
    m.p = 6 * units.g
    self.assertEqual(value(m.p), 6)
    m.p = 7 * units.kg
    self.assertEqual(value(m.p), 7000)
    with self.assertRaises(UnitsError):
        m.p = 1 * units.s
    out = StringIO()
    m.pprint(ostream=out)
    self.assertEqual(out.getvalue().strip(), '\n1 Param Declarations\n    p : Size=1, Index=None, Domain=Any, Default=None, Mutable=True, Units=g\n        Key  : Value\n        None : 7000.0\n\n1 Declarations: p\n        '.strip())