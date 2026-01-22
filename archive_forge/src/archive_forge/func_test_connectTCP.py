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
def test_connectTCP(self):
    """
        L{ClientCreator.connectTCP} calls C{reactor.connectTCP} with the host
        and port information passed to it, and with a factory which will
        construct the protocol passed to L{ClientCreator.__init__}.
        """

    def check(reactor, cc):
        cc.connectTCP('example.com', 1234, 4321, ('1.2.3.4', 9876))
        host, port, factory, timeout, bindAddress = reactor.tcpClients.pop()
        self.assertEqual(host, 'example.com')
        self.assertEqual(port, 1234)
        self.assertEqual(timeout, 4321)
        self.assertEqual(bindAddress, ('1.2.3.4', 9876))
        return factory
    self._basicConnectTest(check)