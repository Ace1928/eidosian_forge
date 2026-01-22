import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_deferred_error(self):

    class SomeTest(TestCase):

        def test_something(self):
            return defer.maybeDeferred(lambda: 1 / 0)
    test = SomeTest('test_something')
    runner = self.make_runner(test)
    result = self.make_result()
    runner.run(result)
    self.assertThat([event[:2] for event in result._events], Equals([('startTest', test), ('addError', test), ('stopTest', test)]))
    error = result._events[1][2]
    self.assertThat(error, KeysEqual('traceback', 'twisted-log'))