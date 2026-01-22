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
def test_domain_set_initializer(self):
    m = ConcreteModel()
    m.I = Set(initialize=[1, 2, 3])
    param_vals = {1: 1, 2: 1, 3: -1}
    m.p = Param(m.I, initialize=param_vals, domain={-1, 1})
    self.assertIsInstance(m.p.domain, _SetData)