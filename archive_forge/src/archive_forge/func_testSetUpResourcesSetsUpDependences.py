import unittest
import testtools
import testresources
from testresources.tests import ResultWithResourceExtensions
def testSetUpResourcesSetsUpDependences(self):
    resource = MockResourceInstance()
    self.resource_manager = MockResource(resource)
    self.resourced_case.resources = [('foo', self.resource_manager)]
    self.resource_manager.resources.append(('bar', MockResource('bar_resource')))
    testresources.setUpResources(self.resourced_case, self.resourced_case.resources, None)
    self.assertEqual(resource, self.resourced_case.foo)
    self.assertEqual('bar_resource', self.resourced_case.foo.bar)