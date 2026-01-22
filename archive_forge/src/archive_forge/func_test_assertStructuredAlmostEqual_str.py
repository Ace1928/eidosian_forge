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
def test_assertStructuredAlmostEqual_str(self):
    self.assertStructuredAlmostEqual('hi', 'hi')
    with self.assertRaisesRegex(self.failureException, "'hi' !~= 'hello'"):
        self.assertStructuredAlmostEqual('hi', 'hello')
    with self.assertRaisesRegex(self.failureException, "'hi' !~= \\['h',"):
        self.assertStructuredAlmostEqual('hi', ['h', 'i'])