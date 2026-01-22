from typing import Optional
from unittest import mock
from twisted.internet import address
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.trial import unittest
from .._wrapper import HAProxyWrappingFactory
def test_overflowBytesSentToWrappedProtocolAfter(self) -> None:
    """
        Test if wrapper writes all data to wrapped protocol after parsing.
        """
    factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
    proto = factory.buildProtocol(address.IPv6Address('TCP', '::1', 8080))
    transport = StringTransportWithDisconnection()
    proto.makeConnection(transport)
    proto.dataReceived(b'PROXY TCP6 ::1 ::2 ')
    proto.dataReceived(b'8080 8888\r\nHTTP/1.1 / GET')
    proto.dataReceived(b'\r\n\r\n')
    self.assertEqual(proto.wrappedProtocol.data, b'HTTP/1.1 / GET\r\n\r\n')