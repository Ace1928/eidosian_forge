import unittest
import testtools
import testresources
from testresources.tests import ResultWithResourceExtensions
def testTearDownResourcesDeletesResourceAttributes(self):
    self.resourced_case.resources = [('foo', self.resource_manager)]
    self.resourced_case.setUpResources()
    self.resourced_case.tearDownResources()
    self.failIf(hasattr(self.resourced_case, 'foo'))