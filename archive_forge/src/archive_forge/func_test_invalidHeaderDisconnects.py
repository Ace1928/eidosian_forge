from typing import Optional
from unittest import mock
from twisted.internet import address
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.trial import unittest
from .._wrapper import HAProxyWrappingFactory
def test_invalidHeaderDisconnects(self) -> None:
    """
        Test if invalid headers result in connectionLost events.
        """
    factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
    proto = factory.buildProtocol(address.IPv6Address('TCP', '::1', 8080))
    transport = StringTransportWithDisconnection()
    transport.protocol = proto
    proto.makeConnection(transport)
    proto.dataReceived(b'\x00' + self.IPV4HEADER[1:])
    self.assertFalse(transport.connected)