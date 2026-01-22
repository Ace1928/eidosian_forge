import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_calls_setUp_test_tearDown_in_sequence(self):
    call_log = []
    a = defer.Deferred()
    a.addCallback(lambda x: call_log.append('a'))
    b = defer.Deferred()
    b.addCallback(lambda x: call_log.append('b'))
    c = defer.Deferred()
    c.addCallback(lambda x: call_log.append('c'))

    class SomeCase(TestCase):

        def setUp(self):
            super().setUp()
            call_log.append('setUp')
            return a

        def test_success(self):
            call_log.append('test')
            return b

        def tearDown(self):
            super().tearDown()
            call_log.append('tearDown')
            return c
    test = SomeCase('test_success')
    timeout = self.make_timeout()
    runner = self.make_runner(test, timeout)
    result = self.make_result()
    reactor = self.make_reactor()

    def fire_a():
        self.assertThat(call_log, Equals(['setUp']))
        a.callback(None)

    def fire_b():
        self.assertThat(call_log, Equals(['setUp', 'a', 'test']))
        b.callback(None)

    def fire_c():
        self.assertThat(call_log, Equals(['setUp', 'a', 'test', 'b', 'tearDown']))
        c.callback(None)
    reactor.callLater(timeout * 0.25, fire_a)
    reactor.callLater(timeout * 0.5, fire_b)
    reactor.callLater(timeout * 0.75, fire_c)
    runner.run(result)
    self.assertThat(call_log, Equals(['setUp', 'a', 'test', 'b', 'tearDown', 'c']))