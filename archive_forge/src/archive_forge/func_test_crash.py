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
def test_crash(self) -> None:
    """
        C{reactor.crash()} stops the reactor and does not fire shutdown
        triggers.
        """
    reactor = self.buildReactor()
    events = []
    reactor.addSystemEventTrigger('before', 'shutdown', lambda: events.append(('before', 'shutdown')))
    reactor.callWhenRunning(reactor.callLater, 0, reactor.crash)
    self.runReactor(reactor)
    testCase = cast(SynchronousTestCase, self)
    testCase.assertFalse(reactor.running)
    testCase.assertFalse(events, 'Shutdown triggers invoked but they should not have been.')