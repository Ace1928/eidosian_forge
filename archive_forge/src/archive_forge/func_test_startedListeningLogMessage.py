import socket
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import defer, error
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import DatagramProtocol
from twisted.internet.test.connectionmixins import LogObserverMixin, findFreePort
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python import context
from twisted.python.log import ILogContext, err
from twisted.test.test_udp import GoodClient, Server
from twisted.trial.unittest import SkipTest
def test_startedListeningLogMessage(self):
    """
        When a port starts, a message including a description of the associated
        protocol is logged.
        """
    loggedMessages = self.observe()
    reactor = self.buildReactor()

    @implementer(ILoggingContext)
    class SomeProtocol(DatagramProtocol):

        def logPrefix(self):
            return 'Crazy Protocol'
    protocol = SomeProtocol()
    p = self.getListeningPort(reactor, protocol)
    expectedMessage = 'Crazy Protocol starting on %d' % (p.getHost().port,)
    self.assertEqual((expectedMessage,), loggedMessages[0]['message'])