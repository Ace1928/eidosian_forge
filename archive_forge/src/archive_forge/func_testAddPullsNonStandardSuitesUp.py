import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testAddPullsNonStandardSuitesUp(self):
    case1 = self.makeTestCase()
    case2 = self.makeTestCase()
    inner_suite = CustomSuite([case1, case2])
    self.optimising_suite.addTest(unittest.TestSuite([unittest.TestSuite([inner_suite])]))
    self.assertEqual([CustomSuite([case1]), CustomSuite([case2])], self.optimising_suite._tests)