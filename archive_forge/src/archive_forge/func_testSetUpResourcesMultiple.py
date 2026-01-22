import unittest
import testtools
import testresources
from testresources.tests import ResultWithResourceExtensions
def testSetUpResourcesMultiple(self):
    self.resourced_case.resources = [('foo', self.resource_manager), ('bar', MockResource('bar_resource'))]
    testresources.setUpResources(self.resourced_case, self.resourced_case.resources, None)
    self.assertEqual(self.resource, self.resourced_case.foo)
    self.assertEqual('bar_resource', self.resourced_case.bar)