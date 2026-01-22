from io import StringIO
import sys
import logging
import pyomo.common.unittest as unittest
from pyomo.contrib.trustregion.util import IterationLogger, minIgnoreNone, maxIgnoreNone
from pyomo.common.log import LoggingIntercept
def test_logIteration(self):
    self.iterLogger.newIteration(self.iteration, self.thetak, self.objk, self.radius, self.stepNorm)
    OUTPUT = StringIO()
    with LoggingIntercept(OUTPUT, 'pyomo.contrib.trustregion', logging.INFO):
        self.iterLogger.logIteration()
    self.assertIn('Iteration 0', OUTPUT.getvalue())
    self.assertIn('feasibility =', OUTPUT.getvalue())
    self.assertIn('stepNorm =', OUTPUT.getvalue())