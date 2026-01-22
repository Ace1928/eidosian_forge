import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testDirtiedResourceNotRecreated(self):
    make_counter = MakeCounter()

    def dirtyResource(test):
        make_counter.dirtied(test._default)
    case = self.makeResourcedTestCase(make_counter, dirtyResource)
    self.optimising_suite.addTest(case)
    result = unittest.TestResult()
    self.optimising_suite.run(result)
    self.assertEqual(result.testsRun, 1)
    self.assertEqual(result.wasSuccessful(), True)
    self.assertEqual(make_counter.makes, 1)