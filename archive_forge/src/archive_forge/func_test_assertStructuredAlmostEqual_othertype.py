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
def test_assertStructuredAlmostEqual_othertype(self):
    a = datetime.datetime(1, 1, 1)
    b = datetime.datetime(1, 1, 1)
    self.assertStructuredAlmostEqual(a, b)
    b = datetime.datetime(1, 1, 2)
    with self.assertRaisesRegex(self.failureException, 'datetime.* !~= datetime'):
        self.assertStructuredAlmostEqual(a, b)