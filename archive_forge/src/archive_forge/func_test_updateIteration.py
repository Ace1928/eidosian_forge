from io import StringIO
import sys
import logging
import pyomo.common.unittest as unittest
from pyomo.contrib.trustregion.util import IterationLogger, minIgnoreNone, maxIgnoreNone
from pyomo.common.log import LoggingIntercept
def test_updateIteration(self):
    self.iterLogger.newIteration(self.iteration, self.thetak, self.objk, self.radius, self.stepNorm)
    self.assertEqual(self.iterLogger.iterations[0].objectiveValue, self.objk)
    self.assertEqual(self.iterLogger.iterations[0].feasibility, self.thetak)
    self.assertEqual(self.iterLogger.iterations[0].trustRadius, self.radius)
    self.assertEqual(self.iterLogger.iterations[0].stepNorm, self.stepNorm)
    self.iterLogger.updateIteration(feasibility=5.0)
    self.assertEqual(self.iterLogger.iterations[0].objectiveValue, self.objk)
    self.assertEqual(self.iterLogger.iterations[0].feasibility, 5.0)
    self.assertEqual(self.iterLogger.iterations[0].trustRadius, self.radius)
    self.assertEqual(self.iterLogger.iterations[0].stepNorm, self.stepNorm)
    self.iterLogger.updateIteration(objectiveValue=0.1)
    self.assertEqual(self.iterLogger.iterations[0].objectiveValue, 0.1)
    self.assertEqual(self.iterLogger.iterations[0].feasibility, 5.0)
    self.assertEqual(self.iterLogger.iterations[0].trustRadius, self.radius)
    self.assertEqual(self.iterLogger.iterations[0].stepNorm, self.stepNorm)
    self.iterLogger.updateIteration(trustRadius=100)
    self.assertEqual(self.iterLogger.iterations[0].objectiveValue, 0.1)
    self.assertEqual(self.iterLogger.iterations[0].feasibility, 5.0)
    self.assertEqual(self.iterLogger.iterations[0].trustRadius, 100)
    self.assertEqual(self.iterLogger.iterations[0].stepNorm, self.stepNorm)
    self.iterLogger.updateIteration(stepNorm=1)
    self.assertEqual(self.iterLogger.iterations[0].objectiveValue, 0.1)
    self.assertEqual(self.iterLogger.iterations[0].feasibility, 5.0)
    self.assertEqual(self.iterLogger.iterations[0].trustRadius, 100)
    self.assertEqual(self.iterLogger.iterations[0].stepNorm, 1)
    self.iterLogger.updateIteration(feasibility=10.0, objectiveValue=0.2, trustRadius=1000, stepNorm=10)
    self.assertEqual(self.iterLogger.iterations[0].objectiveValue, 0.2)
    self.assertEqual(self.iterLogger.iterations[0].feasibility, 10.0)
    self.assertEqual(self.iterLogger.iterations[0].trustRadius, 1000)
    self.assertEqual(self.iterLogger.iterations[0].stepNorm, 10)