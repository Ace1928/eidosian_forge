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
def test_cancelConnectSSLFailedTimeout(self):
    """
        Similar to L{test_cancelConnectSSLTimeout}, but for the case where the
        connection attempt fails.
        """

    def connect(reactor, cc):
        d = cc.connectSSL('example.com', 1234, object())
        host, port, factory, contextFactory, timeout, bindADdress = reactor.sslClients.pop()
        return (d, factory)
    return self._cancelConnectFailedTimeoutTest(connect)