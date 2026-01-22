from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testDirtyingWhenUnused(self):
    resource_manager = MockResource()
    resource = resource_manager.getResource()
    resource_manager.finishedWith(resource)
    resource_manager.dirtied(resource)
    self.assertEqual(1, resource_manager.makes)
    resource = resource_manager.getResource()
    self.assertEqual(2, resource_manager.makes)