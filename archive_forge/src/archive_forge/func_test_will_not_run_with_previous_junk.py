import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_will_not_run_with_previous_junk(self):
    from twisted.internet.protocol import ServerFactory
    reactor = self.make_reactor()
    spinner = self.make_spinner(reactor)
    timeout = self.make_timeout()
    spinner.run(timeout, reactor.listenTCP, 0, ServerFactory(), interface='127.0.0.1')
    self.assertThat(lambda: spinner.run(timeout, lambda: None), Raises(MatchesException(_spinner.StaleJunkError)))