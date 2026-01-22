from typing import Optional
from unittest import mock
from twisted.internet import address
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import StringTransportWithDisconnection
from twisted.trial import unittest
from .._wrapper import HAProxyWrappingFactory
def test_preDataReceived_getPeerHost(self) -> None:
    """
        Before any data is received the HAProxy protocol will return the same peer
        and host as the IP connection.
        """
    factory = HAProxyWrappingFactory(Factory.forProtocol(StaticProtocol))
    proto = factory.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 8080))
    transport = StringTransportWithDisconnection(hostAddress=mock.sentinel.host_address, peerAddress=mock.sentinel.peer_address)
    proto.makeConnection(transport)
    self.assertEqual(proto.getHost(), mock.sentinel.host_address)
    self.assertEqual(proto.getPeer(), mock.sentinel.peer_address)