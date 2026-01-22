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
def test_shutdownFiresTriggersAsynchronously(self) -> None:
    """
        C{"before"} C{"shutdown"} triggers are not run synchronously from
        L{reactor.stop}.
        """
    reactor = self.buildReactor()
    events: List[str] = []
    reactor.addSystemEventTrigger('before', 'shutdown', events.append, 'before shutdown')

    def stopIt() -> None:
        reactor.stop()
        events.append('stopped')
    testCase = cast(SynchronousTestCase, self)
    reactor.callWhenRunning(stopIt)
    testCase.assertEqual(events, [])
    self.runReactor(reactor)
    testCase.assertEqual(events, ['stopped', 'before shutdown'])