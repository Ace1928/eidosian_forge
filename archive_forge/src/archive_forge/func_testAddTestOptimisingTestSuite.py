import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testAddTestOptimisingTestSuite(self):
    case = self.makeTestCase()
    suite1 = testresources.OptimisingTestSuite([case])
    suite2 = testresources.OptimisingTestSuite([case])
    self.optimising_suite.addTest(suite1)
    self.optimising_suite.addTest(suite2)
    self.assertEqual([case, case], self.optimising_suite._tests)