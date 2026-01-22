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
def test_timeout_skip_fails(self):
    try:
        with self.assertRaisesRegex(unittest.SkipTest, 'Skipping this test'):
            self.test_timeout_skip()
        TestPyomoUnittest.test_timeout_skip.skip = False
        with self.assertRaisesRegex(AssertionError, '0 != 1'):
            self.test_timeout_skip()
    finally:
        TestPyomoUnittest.test_timeout_skip.skip = True