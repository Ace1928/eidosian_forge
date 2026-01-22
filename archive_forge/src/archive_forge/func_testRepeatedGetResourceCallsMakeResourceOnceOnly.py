from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testRepeatedGetResourceCallsMakeResourceOnceOnly(self):
    resource_manager = MockResource()
    resource_manager.getResource()
    resource_manager.getResource()
    self.assertEqual(1, resource_manager.makes)