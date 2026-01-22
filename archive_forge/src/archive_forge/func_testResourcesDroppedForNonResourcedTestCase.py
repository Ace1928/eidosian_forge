import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testResourcesDroppedForNonResourcedTestCase(self):
    sample_resource = MakeCounter()

    def resourced_case_hook(test):
        self.assertTrue(sample_resource._uses > 0)
    self.optimising_suite.addTest(self.makeResourcedTestCase(sample_resource, resourced_case_hook))

    def normal_case_hook(test):
        self.assertEqual(sample_resource._uses, 0)
    self.optimising_suite.addTest(self.makeTestCase(normal_case_hook))
    result = unittest.TestResult()
    self.optimising_suite.run(result)
    self.assertEqual(result.testsRun, 2)
    self.assertEqual([], result.failures)
    self.assertEqual([], result.errors)
    self.assertEqual(result.wasSuccessful(), True)