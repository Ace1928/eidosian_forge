from twisted.internet import interfaces
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import TCPCreator
from twisted.internet.test.test_tls import (
from twisted.trial import unittest
from zope.interface import implementer
def startTLSAfterRegisterProducer(self, streaming):
    """
        When a producer is registered, and then startTLS is called,
        the producer is re-registered with the C{TLSMemoryBIOProtocol}.
        """
    clientContext = self.getClientContext()
    serverContext = self.getServerContext()
    result = []
    producer = FakeProducer()

    class RegisterTLSProtocol(ConnectableProtocol):

        def connectionMade(self):
            self.transport.registerProducer(producer, streaming)
            self.transport.startTLS(serverContext)
            if streaming:
                result.append(self.transport.protocol._producer._producer)
                result.append(self.transport.producer._producer)
            else:
                result.append(self.transport.protocol._producer._producer._producer)
                result.append(self.transport.producer._producer._producer)
            self.transport.unregisterProducer()
            self.transport.loseConnection()

    class StartTLSProtocol(ConnectableProtocol):

        def connectionMade(self):
            self.transport.startTLS(clientContext)
    runProtocolsWithReactor(self, RegisterTLSProtocol(), StartTLSProtocol(), TCPCreator())
    self.assertEqual(result, [producer, producer])