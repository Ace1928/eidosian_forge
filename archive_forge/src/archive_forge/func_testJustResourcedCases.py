import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testJustResourcedCases(self):
    resourced_case = self.makeResourcedTestCase()
    resource = resourced_case.resources[0][1]
    resource_set_tests = split_by_resources([resourced_case])
    self.assertEqual({frozenset(): [], frozenset([resource]): [resourced_case]}, resource_set_tests)