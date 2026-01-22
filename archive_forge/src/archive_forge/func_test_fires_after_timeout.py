import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_fires_after_timeout(self):
    reactor = self.make_reactor()
    spinner1 = self.make_spinner(reactor)
    timeout = self.make_timeout()
    deferred1 = defer.Deferred()
    self.expectThat(lambda: spinner1.run(timeout, lambda: deferred1), Raises(MatchesException(_spinner.TimeoutError)))
    marker = object()
    deferred2 = defer.Deferred()
    deferred1.addCallback(lambda ignored: reactor.callLater(0, deferred2.callback, marker))

    def fire_other():
        """Fire Deferred from the last spin while waiting for this one."""
        deferred1.callback(object())
        return deferred2
    spinner2 = self.make_spinner(reactor)
    self.assertThat(spinner2.run(timeout, fire_other), Is(marker))