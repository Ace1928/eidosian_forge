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
def validateDict(self, ref, test):
    test = dict(test)
    ref = dict(ref)
    self.assertEqual(len(test), len(ref))
    for key in test.keys():
        self.assertTrue(key in ref)
        if ref[key] is None:
            self.assertTrue(test[key] is None or test[key].value is None)
        else:
            self.assertEqual(ref[key], value(test[key]))