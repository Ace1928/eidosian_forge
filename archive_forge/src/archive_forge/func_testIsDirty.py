from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testIsDirty(self):
    resource_manager = MockResource()
    r = resource_manager.getResource()
    resource_manager.dirtied(r)
    self.assertTrue(resource_manager.isDirty())
    resource_manager.finishedWith(r)