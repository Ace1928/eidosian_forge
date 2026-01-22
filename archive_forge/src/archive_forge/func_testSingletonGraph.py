import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testSingletonGraph(self):
    resource = self.makeResource()
    suite = testresources.OptimisingTestSuite()
    graph = suite._getGraph([frozenset()])
    self.assertEqual({frozenset(): {}}, graph)