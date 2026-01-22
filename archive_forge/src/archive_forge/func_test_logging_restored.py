import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_logging_restored(self):
    from testtools.twistedsupport._runtest import _NoTwistedLogObservers

    class SomeTest(TestCase):

        def test_something(self):
            self.useFixture(_NoTwistedLogObservers())
            log.msg('foo')

    def run_then_log():
        SomeTest('test_something').run()
        log.msg('bar')
    _, messages = self._get_logged_messages(run_then_log)
    self.assertThat(messages, MatchesListwise([ContainsDict({'message': Equals(('bar',))})]))