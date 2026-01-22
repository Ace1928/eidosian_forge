from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testResetActivityForResourceWithoutExtensions(self):
    result = ResultWithoutResourceExtensions()
    resource_manager = MockResource()
    resource_manager.getResource()
    r = resource_manager.getResource()
    resource_manager.dirtied(r)
    resource_manager.finishedWith(r)
    r = resource_manager.getResource(result)
    resource_manager.dirtied(r)
    resource_manager.finishedWith(r)
    resource_manager.finishedWith(resource_manager._currentResource)