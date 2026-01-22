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
def test_stopWhenAlreadyStopped(self) -> None:
    """
        C{reactor.stop()} raises L{RuntimeError} when called after the reactor
        has been stopped.
        """
    reactor = self.buildReactor()
    reactor.callWhenRunning(reactor.stop)
    self.runReactor(reactor)
    cast(SynchronousTestCase, self).assertRaises(RuntimeError, reactor.stop)