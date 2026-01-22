from twisted.python.reflect import requireModule
from twisted.internet.address import IPv6Address
from twisted.internet.test.test_endpoints import deterministicResolvingReactor
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.trial import unittest
def test_channelOpenHostnameRequests(self) -> None:
    """
        When a hostname is sent as part of forwarding requests, it
        is resolved using HostnameEndpoint's resolver.
        """
    sut = forwarding.SSHConnectForwardingChannel(hostport=('fwd.example.org', 1234))
    memoryReactor = MemoryReactorClock()
    sut._reactor = deterministicResolvingReactor(memoryReactor, ['::1'])
    sut.channelOpen(None)
    self.makeTCPConnection(memoryReactor)
    self.successResultOf(sut._channelOpenDeferred)
    self.assertIsInstance(sut.client, forwarding.SSHForwardingClient)
    self.assertEqual(IPv6Address('TCP', '::1', 1234), sut.client.transport.getPeer())