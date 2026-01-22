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
def test_assertStructuredAlmostEqual_errorChecking(self):
    with self.assertRaisesRegex(ValueError, 'Cannot specify more than one of {places, delta, abstol}'):
        self.assertStructuredAlmostEqual(1, 1, places=7, delta=1)