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
def test_timeout_fcn_call(self):
    self.assertEqual(short_sleep(), 42)
    with self.assertRaisesRegex(TimeoutError, 'test timed out after 0.01 seconds'):
        long_sleep()
    with self.assertRaisesRegex(NameError, "name 'foo' is not defined\\s+Original traceback:"):
        raise_exception()
    with self.assertRaisesRegex(AssertionError, '^0 != 1$'):
        fail()