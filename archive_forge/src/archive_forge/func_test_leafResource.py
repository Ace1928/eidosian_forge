from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_leafResource(self) -> None:
    """
        L{getChildForRequest} returns the first resource it encounters with a
        C{isLeaf} attribute set to C{True}.
        """
    request = DummyRequest([b'foo', b'bar'])
    resource = Resource()
    resource.isLeaf = True
    result = getChildForRequest(resource, request)
    self.assertIdentical(resource, result)