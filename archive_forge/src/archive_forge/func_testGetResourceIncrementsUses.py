from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testGetResourceIncrementsUses(self):
    resource_manager = MockResource()
    resource_manager.getResource()
    self.assertEqual(1, resource_manager._uses)
    resource_manager.getResource()
    self.assertEqual(2, resource_manager._uses)