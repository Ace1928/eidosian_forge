import unittest
import testtools
import testresources
from testresources.tests import ResultWithResourceExtensions
def testTearDownResourcesStopsUsingDependencies(self):
    resource = MockResourceInstance()
    dep1 = MockResource('bar_resource')
    self.resource_manager = MockResource(resource)
    self.resourced_case.resources = [('foo', self.resource_manager)]
    self.resource_manager.resources.append(('bar', dep1))
    self.resourced_case.setUpResources()
    self.resourced_case.tearDownResources()
    self.assertEqual(dep1._uses, 0)