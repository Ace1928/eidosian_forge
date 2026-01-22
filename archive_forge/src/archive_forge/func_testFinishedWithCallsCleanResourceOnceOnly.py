from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testFinishedWithCallsCleanResourceOnceOnly(self):
    resource_manager = MockResource()
    resource = resource_manager.getResource()
    resource = resource_manager.getResource()
    resource_manager.finishedWith(resource)
    self.assertEqual(0, resource_manager.cleans)
    resource_manager.finishedWith(resource)
    self.assertEqual(1, resource_manager.cleans)