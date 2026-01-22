import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testOptimisedRunNonResourcedTestCase(self):
    case = self.makeTestCase()
    self.optimising_suite.addTest(case)
    result = unittest.TestResult()
    self.optimising_suite.run(result)
    self.assertEqual(result.testsRun, 1)
    self.assertEqual(result.wasSuccessful(), True)