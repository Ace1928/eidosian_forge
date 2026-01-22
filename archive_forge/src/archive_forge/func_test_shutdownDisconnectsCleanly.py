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
def test_shutdownDisconnectsCleanly(self) -> None:
    """
        A L{IFileDescriptor.connectionLost} implementation which raises an
        exception does not prevent the remaining L{IFileDescriptor}s from
        having their C{connectionLost} method called.
        """
    lostOK = [False]

    class ProblematicFileDescriptor(FileDescriptor):

        def connectionLost(self, reason: Failure) -> None:
            raise RuntimeError('simulated connectionLost error')

    class OKFileDescriptor(FileDescriptor):

        def connectionLost(self, reason: Failure) -> None:
            lostOK[0] = True
    testCase = cast(SynchronousTestCase, self)
    reactor = self.buildReactor()
    fds = iter([ProblematicFileDescriptor(), OKFileDescriptor()])
    reactor.removeAll = lambda: fds
    reactor.callWhenRunning(reactor.stop)
    self.runReactor(reactor)
    testCase.assertEqual(len(testCase.flushLoggedErrors(RuntimeError)), 1)
    testCase.assertTrue(lostOK[0])