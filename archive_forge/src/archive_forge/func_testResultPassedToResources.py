import unittest
import testtools
import testresources
from testresources.tests import ResultWithResourceExtensions
def testResultPassedToResources(self):
    result = ResultWithResourceExtensions()
    self.resourced_case.resources = [('foo', self.resource_manager)]
    self.resourced_case.run(result)
    self.assertEqual(4, len(result._calls))