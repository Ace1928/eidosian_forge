from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def test_contentLength(self):
    """
        If the response contains a I{Content-Length} header, the inbound
        request object should still only have C{finish} called on it once.
        """
    data = b'foo bar baz'
    return self._testDataForward(200, b'OK', [(b'Content-Length', [str(len(data)).encode('ascii')])], data)