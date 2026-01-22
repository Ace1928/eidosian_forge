import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_async_cleanups(self):

    class SomeCase(TestCase):

        def test_whatever(self):
            pass
    test = SomeCase('test_whatever')
    call_log = []
    a = defer.Deferred().addCallback(lambda x: call_log.append('a'))
    b = defer.Deferred().addCallback(lambda x: call_log.append('b'))
    c = defer.Deferred().addCallback(lambda x: call_log.append('c'))
    test.addCleanup(lambda: a)
    test.addCleanup(lambda: b)
    test.addCleanup(lambda: c)

    def fire_a():
        self.assertThat(call_log, Equals([]))
        a.callback(None)

    def fire_b():
        self.assertThat(call_log, Equals(['a']))
        b.callback(None)

    def fire_c():
        self.assertThat(call_log, Equals(['a', 'b']))
        c.callback(None)
    timeout = self.make_timeout()
    reactor = self.make_reactor()
    reactor.callLater(timeout * 0.25, fire_a)
    reactor.callLater(timeout * 0.5, fire_b)
    reactor.callLater(timeout * 0.75, fire_c)
    runner = self.make_runner(test, timeout)
    result = self.make_result()
    runner.run(result)
    self.assertThat(call_log, Equals(['a', 'b', 'c']))