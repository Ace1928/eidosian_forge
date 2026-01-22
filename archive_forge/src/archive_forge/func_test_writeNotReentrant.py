from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def test_writeNotReentrant(self):
    """
        L{loopback.loopbackAsync} does not call a protocol's C{dataReceived}
        method while that protocol's transport's C{write} method is higher up
        on the stack.
        """

    class Server(Protocol):

        def dataReceived(self, bytes):
            self.transport.write(b'bytes')

    class Client(Protocol):
        ready = False

        def connectionMade(self):
            reactor.callLater(0, self.go)

        def go(self):
            self.transport.write(b'foo')
            self.ready = True

        def dataReceived(self, bytes):
            self.wasReady = self.ready
            self.transport.loseConnection()
    server = Server()
    client = Client()
    d = loopback.loopbackAsync(client, server)

    def cbFinished(ignored):
        self.assertTrue(client.wasReady)
    d.addCallback(cbFinished)
    return d