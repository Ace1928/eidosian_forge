from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def test_processWithPort(self):
    """
        Check that L{ProxyRequest.process} correctly parse port in the incoming
        URL, and create an outgoing connection with this port.
        """
    transport = StringTransportWithDisconnection()
    channel = DummyChannel(transport)
    reactor = MemoryReactor()
    request = ProxyRequest(channel, False, reactor)
    request.gotLength(0)
    request.requestReceived(b'GET', b'http://example.com:1234/foo/bar', b'HTTP/1.0')
    self.assertEqual(len(reactor.tcpClients), 1)
    self.assertEqual(reactor.tcpClients[0][0], 'example.com')
    self.assertEqual(reactor.tcpClients[0][1], 1234)