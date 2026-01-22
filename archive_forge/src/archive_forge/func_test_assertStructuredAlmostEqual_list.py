import datetime
import multiprocessing
import os
import time
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import ConcreteModel, Var, Param
def test_assertStructuredAlmostEqual_list(self):
    a = [1, 2]
    b = [1, 2, 3]
    with self.assertRaisesRegex(self.failureException, 'sequences are different sizes \\(2 != 3\\)'):
        self.assertStructuredAlmostEqual(a, b)
    self.assertStructuredAlmostEqual(a, b, allow_second_superset=True)
    a.append(3)
    self.assertStructuredAlmostEqual(a, b)
    b[1] -= 1.999e-07
    self.assertStructuredAlmostEqual(a, b)
    b[1] -= 1.999e-07
    with self.assertRaisesRegex(self.failureException, '2 !~= 1.999'):
        self.assertStructuredAlmostEqual(a, b)