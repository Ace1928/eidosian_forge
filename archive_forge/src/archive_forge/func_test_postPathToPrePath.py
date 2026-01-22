from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_postPathToPrePath(self) -> None:
    """
        As path segments from the request are traversed, they are taken from
        C{postpath} and put into C{prepath}.
        """
    request = DummyRequest([b'foo', b'bar'])
    root = Resource()
    child = Resource()
    child.isLeaf = True
    root.putChild(b'foo', child)
    self.assertIdentical(child, getChildForRequest(root, request))
    self.assertEqual(request.prepath, [b'foo'])
    self.assertEqual(request.postpath, [b'bar'])