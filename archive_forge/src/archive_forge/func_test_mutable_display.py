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
def test_mutable_display(self):
    model = ConcreteModel()
    model.P = Param([1, 2], default=0.0, mutable=True)
    model.Q = Param([1, 2], initialize=0.0, mutable=True)
    model.R = Param([1, 2], mutable=True)
    model.R[1] = 0.0
    model.R[2] = 0.0
    for Item in [model.P]:
        f = StringIO()
        display(Item, f)
        tmp = f.getvalue().splitlines()
        self.assertEqual(len(tmp), 2)
    for Item in [model.Q, model.R]:
        f = StringIO()
        display(Item, f)
        tmp = f.getvalue().splitlines()
        for tmp_ in tmp[2:]:
            val = float(tmp_.split(':')[-1].strip())
            self.assertEqual(0, val)
    for Item in [model.P, model.Q, model.R]:
        for i in [1, 2]:
            self.assertEqual(Item[i].value, 0.0)
    for Item in [model.P, model.Q, model.R]:
        f = StringIO()
        display(Item, f)
        tmp = f.getvalue().splitlines()
        for tmp_ in tmp[2:]:
            val = float(tmp_.split(':')[-1].strip())
            self.assertEqual(0, val)
    model.P[1] = 1.0
    model.P[2] = 2.0
    model.Q[1] = 1.0
    model.Q[2] = 2.0
    model.R[1] = 1.0
    model.R[2] = 2.0
    for Item in [model.P, model.Q, model.R]:
        f = StringIO()
        display(Item, f)
        tmp = f.getvalue().splitlines()
        i = 0
        for tmp_ in tmp[2:]:
            i += 1
            val = float(tmp_.split(':')[-1].strip())
            self.assertEqual(i, val)