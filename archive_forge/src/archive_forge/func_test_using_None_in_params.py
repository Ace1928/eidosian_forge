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
def test_using_None_in_params(self):
    m = ConcreteModel()
    m.p = Param(mutable=True)
    self.assertEqual(len(m.p), 0)
    self.assertEqual(len(m.p._data), 0)
    m.p = None
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 1)
    self.assertIs(m.p.value, None)
    m.p = 1
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 1)
    self.assertEqual(m.p.value, 1)
    m = ConcreteModel()
    m.p = Param(mutable=True, initialize=None)
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 1)
    self.assertIs(m.p.value, None)
    m.p = 1
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 1)
    self.assertEqual(m.p.value, 1)
    m = ConcreteModel()
    m.p = Param(mutable=True, default=None)
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 0)
    self.assertIs(m.p.value, None)
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 1)
    m.p = 1
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 1)
    self.assertEqual(m.p.value, 1)
    m = ConcreteModel()
    m.p = Param(mutable=False, initialize=None)
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 1)
    self.assertIs(m.p.value, None)
    m = ConcreteModel()
    m.p = Param(mutable=False, default=None)
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 0)
    self.assertIs(m.p.value, None)
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 0)
    m = ConcreteModel()
    m.p = Param([1, 2], mutable=True)
    self.assertEqual(len(m.p), 0)
    self.assertEqual(len(m.p._data), 0)
    m.p[1] = None
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 1)
    self.assertIs(m.p[1].value, None)
    m.p[1] = 1
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 1)
    self.assertEqual(m.p[1].value, 1)
    m = ConcreteModel()
    m.p = Param([1, 2], mutable=True, initialize={1: None})
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 1)
    self.assertIs(m.p[1].value, None)
    m.p[2] = 1
    self.assertEqual(len(m.p), 2)
    self.assertEqual(len(m.p._data), 2)
    self.assertEqual(m.p[1].value, None)
    self.assertEqual(m.p[2].value, 1)
    m = ConcreteModel()
    m.p = Param([1, 2], mutable=True, default=None)
    self.assertEqual(len(m.p), 2)
    self.assertEqual(len(m.p._data), 0)
    self.assertIs(m.p[1].value, None)
    self.assertEqual(len(m.p), 2)
    self.assertEqual(len(m.p._data), 1)
    m.p[2] = 1
    self.assertEqual(len(m.p), 2)
    self.assertEqual(len(m.p._data), 2)
    self.assertIs(m.p[1].value, None)
    self.assertEqual(m.p[2].value, 1)
    m = ConcreteModel()
    m.p = Param([1, 2], mutable=False, initialize={1: None})
    self.assertEqual(len(m.p), 1)
    self.assertEqual(len(m.p._data), 1)
    self.assertIs(m.p[1], None)
    m = ConcreteModel()
    m.p = Param([1, 2], mutable=False, default=None)
    self.assertEqual(len(m.p), 2)
    self.assertEqual(len(m.p._data), 0)
    self.assertIs(m.p[1], None)
    self.assertEqual(len(m.p), 2)
    self.assertEqual(len(m.p._data), 0)