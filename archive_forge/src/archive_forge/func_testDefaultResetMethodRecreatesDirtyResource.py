from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testDefaultResetMethodRecreatesDirtyResource(self):
    resource_manager = MockResource()
    resource = resource_manager.getResource()
    self.assertEqual(1, resource_manager.makes)
    resource_manager.dirtied(resource)
    resource_manager.reset(resource)
    self.assertEqual(2, resource_manager.makes)
    self.assertEqual(1, resource_manager.cleans)