import signal
import time
from types import FrameType
from typing import Callable, List, Optional, Tuple, Union, cast
from twisted.internet.abstract import FileDescriptor
from twisted.internet.defer import Deferred
from twisted.internet.error import ReactorAlreadyRunning, ReactorNotRestartable
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
def test_callWhenRunningOrder(self) -> None:
    """
        Functions are run in the order that they were passed to
        L{reactor.callWhenRunning}.
        """
    reactor = self.buildReactor()
    events: List[str] = []
    reactor.callWhenRunning(events.append, 'first')
    reactor.callWhenRunning(events.append, 'second')
    reactor.callWhenRunning(reactor.stop)
    self.runReactor(reactor)
    cast(SynchronousTestCase, self).assertEqual(events, ['first', 'second'])