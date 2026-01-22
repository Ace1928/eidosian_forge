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
@unittest.expectedFailure
@unittest.timeout(0.01)
def test_timeout_timeout(self):
    time.sleep(1)
    self.assertEqual(0, 1)