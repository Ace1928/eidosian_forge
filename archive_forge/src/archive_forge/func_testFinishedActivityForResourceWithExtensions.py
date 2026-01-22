from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testFinishedActivityForResourceWithExtensions(self):
    result = ResultWithResourceExtensions()
    resource_manager = MockResource()
    r = resource_manager.getResource()
    expected = [('clean', 'start', resource_manager), ('clean', 'stop', resource_manager)]
    resource_manager.finishedWith(r, result)
    self.assertEqual(expected, result._calls)