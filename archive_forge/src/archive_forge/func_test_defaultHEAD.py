from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_defaultHEAD(self) -> None:
    """
        When not otherwise overridden, L{Resource.render} treats a I{HEAD}
        request as if it were a I{GET} request.
        """
    expected = b'insert response here'
    request = DummyRequest([])
    request.method = b'HEAD'
    resource = BytesReturnedRenderable(expected)
    self.assertEqual(expected, resource.render(request))