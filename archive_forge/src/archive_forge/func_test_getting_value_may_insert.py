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
def test_getting_value_may_insert(self):
    m = ConcreteModel()
    m.p = Param(mutable=True)
    self.assertFalse(None in m.p)
    m.p.value = None
    self.assertTrue(None in m.p)
    m.q = Param()
    self.assertFalse(None in m.q)
    with self.assertRaises(ValueError):
        m.q.value
    self.assertFalse(None in m.q)
    m.qm = Param(mutable=True)
    self.assertFalse(None in m.qm)
    with self.assertRaises(ValueError):
        m.qm.value
    self.assertTrue(None in m.qm)
    m.r = Param([1], mutable=True)
    self.assertFalse(1 in m.r)
    m.r[1]
    self.assertTrue(1 in m.r)