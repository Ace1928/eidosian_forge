import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testEmptyGraph(self):
    suite = testresources.OptimisingTestSuite()
    graph = suite._getGraph([])
    self.assertEqual({}, graph)