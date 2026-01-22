from twisted.internet.defer import gatherResults
from twisted.trial.unittest import TestCase
from twisted.web.http import NOT_FOUND
from twisted.web.resource import NoResource
from twisted.web.server import Site
from twisted.web.static import Data
from twisted.web.test._util import _render
from twisted.web.test.test_web import DummyRequest
from twisted.web.vhost import NameVirtualHost, VHostMonsterResource, _HostResource
def test_getChild(self):
    """
        L{VHostMonsterResource.getChild} returns I{_HostResource} and modifies
        I{Request} with correct L{Request.isSecure}.
        """
    vhm = VHostMonsterResource()
    request = DummyRequest([])
    self.assertIsInstance(vhm.getChild(b'http', request), _HostResource)
    self.assertFalse(request.isSecure())
    request = DummyRequest([])
    self.assertIsInstance(vhm.getChild(b'https', request), _HostResource)
    self.assertTrue(request.isSecure())