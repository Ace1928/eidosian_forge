import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testNoResources(self):
    self.assertEqual(0, self.suite.cost_of_switching(set(), set()))