import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_only_addError_once(self):
    self.useFixture(DebugTwisted(False))
    reactor = self.make_reactor()

    class WhenItRains(TestCase):

        def it_pours(self):
            self.addCleanup(lambda: 3 / 0)
            from twisted.internet.protocol import ServerFactory
            reactor.listenTCP(0, ServerFactory(), interface='127.0.0.1')
            defer.maybeDeferred(lambda: 2 / 0)
            raise RuntimeError('Excess precipitation')
    test = WhenItRains('it_pours')
    runner = self.make_runner(test)
    result = self.make_result()
    runner.run(result)
    self.assertThat([event[:2] for event in result._events], Equals([('startTest', test), ('addError', test), ('stopTest', test)]))
    error = result._events[1][2]
    self.assertThat(error, KeysEqual('traceback', 'traceback-1', 'traceback-2', 'twisted-log', 'unhandled-error-in-deferred'))