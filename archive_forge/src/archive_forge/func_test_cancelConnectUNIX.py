from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import (
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_cancelConnectUNIX(self):
    """
        The L{Deferred} returned by L{ClientCreator.connectTCP} can be cancelled
        to abort the connection attempt before it completes.
        """

    def connect(cc):
        return cc.connectUNIX('/foo/bar')
    return self._cancelConnectTest(connect)