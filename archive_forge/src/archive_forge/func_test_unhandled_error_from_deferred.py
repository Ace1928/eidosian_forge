import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_unhandled_error_from_deferred(self):
    self.useFixture(DebugTwisted(False))

    class SomeCase(TestCase):

        def test_cruft(self):
            defer.maybeDeferred(lambda: 1 / 0)
            defer.maybeDeferred(lambda: 2 / 0)
    test = SomeCase('test_cruft')
    runner = self.make_runner(test)
    result = self.make_result()
    runner.run(result)
    error = result._events[1][2]
    result._events[1] = ('addError', test, None)
    self.assertThat(result._events, Equals([('startTest', test), ('addError', test, None), ('stopTest', test)]))
    self.assertThat(error, KeysEqual('twisted-log', 'unhandled-error-in-deferred', 'unhandled-error-in-deferred-1'))