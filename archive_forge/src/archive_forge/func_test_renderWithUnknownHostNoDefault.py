from twisted.internet.defer import gatherResults
from twisted.trial.unittest import TestCase
from twisted.web.http import NOT_FOUND
from twisted.web.resource import NoResource
from twisted.web.server import Site
from twisted.web.static import Data
from twisted.web.test._util import _render
from twisted.web.test.test_web import DummyRequest
from twisted.web.vhost import NameVirtualHost, VHostMonsterResource, _HostResource
def test_renderWithUnknownHostNoDefault(self):
    """
        L{NameVirtualHost.render} returns a response with a status of I{NOT
        FOUND} if the instance's C{default} is L{None} and there is no host
        matching the value of the I{Host} header in the request.
        """
    virtualHostResource = NameVirtualHost()
    request = DummyRequest([b''])
    request.requestHeaders.addRawHeader(b'host', b'example.com')
    d = _render(virtualHostResource, request)

    def cbRendered(ignored):
        self.assertEqual(request.responseCode, NOT_FOUND)
    d.addCallback(cbRendered)
    return d