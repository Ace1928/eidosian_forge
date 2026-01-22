import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def test_writeAfterShutdownWithoutReading(self):
    """
        A TCP transport which is written to after the connection has been shut
        down should notify its protocol that the connection has been lost, even
        if the TCP transport is not actively being monitored for read events
        (ie, pauseProducing was called on it).
        """
    if reactor.__class__.__name__ == 'IOCPReactor':
        raise SkipTest('iocpreactor does not, in fact, stop reading immediately after pauseProducing is called. This results in a bonus disconnection notification. Under some circumstances, it might be possible to not receive this notifications (specifically, pauseProducing, deliver some data, proceed with this test). ')
    clientPaused = defer.Deferred()
    serverLost = defer.Deferred()

    class Disconnecter(protocol.Protocol):
        """
            Protocol for the server side of the connection which disconnects
            itself in a callback on clientPaused and publishes notification
            when its connection is actually lost.
            """

        def connectionMade(self):
            """
                Set up a callback on clientPaused to lose the connection.
                """
            msg('Disconnector.connectionMade')

            def disconnect(ignored):
                msg('Disconnector.connectionMade disconnect')
                self.transport.loseConnection()
                msg('loseConnection called')
            clientPaused.addCallback(disconnect)

        def connectionLost(self, reason):
            """
                Notify observers that the server side of the connection has
                ended.
                """
            msg('Disconnecter.connectionLost')
            serverLost.callback(None)
            msg('serverLost called back')
    server = protocol.ServerFactory()
    server.protocol = Disconnecter
    port = reactor.listenTCP(0, server, interface='127.0.0.1')
    self.addCleanup(port.stopListening)
    addr = port.getHost()

    @implementer(IPullProducer)
    class Infinite:
        """
            A producer which will write to its consumer as long as
            resumeProducing is called.

            @ivar consumer: The L{IConsumer} which will be written to.
            """

        def __init__(self, consumer):
            self.consumer = consumer

        def resumeProducing(self):
            msg('Infinite.resumeProducing')
            self.consumer.write(b'x')
            msg('Infinite.resumeProducing wrote to consumer')

        def stopProducing(self):
            msg('Infinite.stopProducing')

    class UnreadingWriter(protocol.Protocol):
        """
            Trivial protocol which pauses its transport immediately and then
            writes some bytes to it.
            """

        def connectionMade(self):
            msg('UnreadingWriter.connectionMade')
            self.transport.pauseProducing()
            clientPaused.callback(None)
            msg('clientPaused called back')

            def write(ignored):
                msg('UnreadingWriter.connectionMade write')
                producer = Infinite(self.transport)
                msg('UnreadingWriter.connectionMade write created producer')
                self.transport.registerProducer(producer, False)
                msg('UnreadingWriter.connectionMade write registered producer')
            serverLost.addCallback(write)
    client = MyClientFactory()
    client.protocolFactory = UnreadingWriter
    clientConnectionLost = client.deferred

    def cbClientLost(ignored):
        msg('cbClientLost')
        return client.lostReason
    clientConnectionLost.addCallback(cbClientLost)
    msg(f'Connecting to {addr.host}:{addr.port}')
    reactor.connectTCP(addr.host, addr.port, client)
    msg('Returning Deferred')
    return self.assertFailure(clientConnectionLost, error.ConnectionLost)