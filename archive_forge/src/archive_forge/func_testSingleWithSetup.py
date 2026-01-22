import unittest
import testtools
import testresources
from testresources.tests import ResultWithResourceExtensions
def testSingleWithSetup(self):
    self.resourced_case.resources = [('foo', self.resource_manager)]
    self.resourced_case.setUp()
    self.assertEqual(self.resourced_case.foo, self.resource)
    self.assertEqual(self.resource_manager._uses, 1)
    self.resourced_case.tearDown()
    self.failIf(hasattr(self.resourced_case, 'foo'))
    self.assertEqual(self.resource_manager._uses, 0)