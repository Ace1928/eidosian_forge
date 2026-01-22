from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testFinishedWithResetsCurrentResource(self):
    resource_manager = MockResource()
    resource = resource_manager.getResource()
    resource_manager.finishedWith(resource)
    self.assertIs(None, resource_manager._currentResource)