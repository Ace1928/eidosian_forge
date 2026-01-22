from io import StringIO
import sys
import logging
import pyomo.common.unittest as unittest
from pyomo.contrib.trustregion.util import IterationLogger, minIgnoreNone, maxIgnoreNone
from pyomo.common.log import LoggingIntercept
def test_printIteration(self):
    self.iterLogger.newIteration(self.iteration, self.thetak, self.objk, self.radius, self.stepNorm)
    OUTPUT = StringIO()
    sys.stdout = OUTPUT
    self.iterLogger.printIteration()
    sys.stdout = sys.__stdout__
    self.assertIn(str(self.radius), OUTPUT.getvalue())
    self.assertIn(str(self.iteration), OUTPUT.getvalue())
    self.assertIn(str(self.thetak), OUTPUT.getvalue())
    self.assertIn(str(self.objk), OUTPUT.getvalue())
    self.assertIn(str(self.stepNorm), OUTPUT.getvalue())