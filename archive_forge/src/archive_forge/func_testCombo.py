import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testCombo(self):
    a = self.makeResource()
    b = self.makeResource()
    c = self.makeResource()
    self.assertEqual(2, self.suite.cost_of_switching(set([a]), set([b])))
    self.assertEqual(2, self.suite.cost_of_switching(set([a, c]), set([b, c])))