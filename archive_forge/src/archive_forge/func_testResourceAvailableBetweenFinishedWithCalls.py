from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testResourceAvailableBetweenFinishedWithCalls(self):
    resource_manager = MockResource()
    resource = resource_manager.getResource()
    resource = resource_manager.getResource()
    resource_manager.finishedWith(resource)
    self.assertIs(resource, resource_manager._currentResource)
    resource_manager.finishedWith(resource)