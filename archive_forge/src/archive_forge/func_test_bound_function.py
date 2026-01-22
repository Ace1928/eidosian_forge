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
def test_bound_function(self):
    if multiprocessing.get_start_method() == 'fork':
        self.bound_function()
        return
    LOG = StringIO()
    with LoggingIntercept(LOG):
        with self.assertRaises((TypeError, EOFError, AttributeError)):
            self.bound_function()
    self.assertIn("platform that does not support 'fork'", LOG.getvalue())
    self.assertIn('one of its arguments is not serializable', LOG.getvalue())