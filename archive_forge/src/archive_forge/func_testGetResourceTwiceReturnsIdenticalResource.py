from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testGetResourceTwiceReturnsIdenticalResource(self):
    resource_manager = MockResource()
    resource1 = resource_manager.getResource()
    resource2 = resource_manager.getResource()
    self.assertIs(resource1, resource2)