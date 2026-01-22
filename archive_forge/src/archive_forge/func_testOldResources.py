import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testOldResources(self):
    a = self.makeResource()
    b = self.makeResource()
    self.assertEqual(1, self.suite.cost_of_switching(set([a]), set()))
    self.assertEqual(1, self.suite.cost_of_switching(set([a, b]), set([a])))
    self.assertEqual(2, self.suite.cost_of_switching(set([a, b]), set()))