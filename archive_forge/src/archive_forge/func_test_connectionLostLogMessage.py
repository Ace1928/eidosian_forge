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
def test_connectionLostLogMessage(self):
    """
        When a connection is lost a message is logged containing an
        address identifying the port and the fact that it was closed.
        """
    loggedMessages = self.observe()
    reactor = self.buildReactor()
    p = self.getListeningPort(reactor, DatagramProtocol())
    expectedMessage = f'(UDP Port {p.getHost().port} Closed)'

    def stopReactor(ignored):
        reactor.stop()

    def doStopListening():
        del loggedMessages[:]
        maybeDeferred(p.stopListening).addCallback(stopReactor)
    reactor.callWhenRunning(doStopListening)
    self.runReactor(reactor)
    self.assertEqual((expectedMessage,), loggedMessages[0]['message'])