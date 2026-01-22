from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testInitiallyNotDirty(self):
    resource_manager = testresources.TestResource()
    self.assertEqual(False, resource_manager._dirty)