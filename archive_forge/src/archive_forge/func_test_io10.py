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
def test_io10(self):
    OUTPUT = open('param.dat', 'w')
    OUTPUT.write('data;\n')
    OUTPUT.write('set A1 := a b c d e f g h i j k l ;\n')
    OUTPUT.write('set A2 := 2 4 6 ;\n')
    OUTPUT.write('param B :=\n')
    OUTPUT.write(' [*,2,*] a b 1 c d 2 e f 3\n')
    OUTPUT.write(' [*,4,*] g h 4 i j 5\n')
    OUTPUT.write(' [*,6,*] k l 6\n')
    OUTPUT.write(';\n')
    OUTPUT.write('end;\n')
    OUTPUT.close()
    self.model.A1 = Set()
    self.model.A2 = Set()
    self.model.B = Param(self.model.A1, self.model.A2, self.model.A1)
    self.instance = self.model.create_instance('param.dat')
    self.assertEqual(set(self.instance.B.sparse_keys()), set([('e', 2, 'f'), ('c', 2, 'd'), ('a', 2, 'b'), ('i', 4, 'j'), ('g', 4, 'h'), ('k', 6, 'l')]))