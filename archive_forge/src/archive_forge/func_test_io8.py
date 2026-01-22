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
def test_io8(self):
    OUTPUT = open('param.dat', 'w')
    OUTPUT.write('data;\n')
    OUTPUT.write('param : A : B :=\n')
    OUTPUT.write('"A" 3.3\n')
    OUTPUT.write('"B" 3.4\n')
    OUTPUT.write('"C" 3.5;\n')
    OUTPUT.write('end;\n')
    OUTPUT.close()
    self.model.A = Set()
    self.model.B = Param(self.model.A)
    self.instance = self.model.create_instance('param.dat')
    self.assertEqual(set(self.instance.A.data()), set(['A', 'B', 'C']))