from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testGetResourceResetsUsedResource(self):
    resource_manager = MockResettableResource()
    resource_manager.getResource()
    resource = resource_manager.getResource()
    self.assertEqual(1, resource_manager.makes)
    resource_manager.dirtied(resource)
    resource_manager.getResource()
    self.assertEqual(1, resource_manager.makes)
    self.assertEqual(1, resource_manager.resets)
    resource_manager.finishedWith(resource)