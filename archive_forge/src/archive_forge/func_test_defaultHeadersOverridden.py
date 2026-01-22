from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def test_defaultHeadersOverridden(self):
    """
        L{server.Request} within the proxy sets certain response headers by
        default. When we get these headers back from the remote server, the
        defaults are overridden rather than simply appended.
        """
    request = self.makeRequest(b'foo')
    request.responseHeaders.setRawHeaders(b'server', [b'old-bar'])
    request.responseHeaders.setRawHeaders(b'date', [b'old-baz'])
    request.responseHeaders.setRawHeaders(b'content-type', [b'old/qux'])
    client = self.makeProxyClient(request, headers={b'accept': b'text/html'})
    self.connectProxy(client)
    headers = {b'Server': [b'bar'], b'Date': [b'2010-01-01'], b'Content-Type': [b'application/x-baz']}
    client.dataReceived(self.makeResponseBytes(200, b'OK', headers.items(), b''))
    self.assertForwardsResponse(request, 200, b'OK', list(headers.items()), b'')