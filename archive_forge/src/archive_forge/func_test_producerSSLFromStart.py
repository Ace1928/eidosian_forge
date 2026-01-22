from twisted.internet import interfaces
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import TCPCreator
from twisted.internet.test.test_tls import (
from twisted.trial import unittest
from zope.interface import implementer
def test_producerSSLFromStart(self):
    """
        C{registerProducer} and C{unregisterProducer} on TLS transports
        created as SSL from the get go are passed to the
        C{TLSMemoryBIOProtocol}, not the underlying transport directly.
        """
    result = []
    producer = FakeProducer()
    runProtocolsWithReactor(self, ConnectableProtocol(), ProducerProtocol(producer, result), SSLCreator())
    self.assertEqual(result, [producer, None])