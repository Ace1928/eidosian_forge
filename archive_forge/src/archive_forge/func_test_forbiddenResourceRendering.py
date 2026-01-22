from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_forbiddenResourceRendering(self) -> None:
    """
        L{ForbiddenResource} sets the HTTP I{FORBIDDEN} code.
        """
    detail = 'longer message'
    page = self.forbiddenResource(detail)
    self._pageRenderingTest(page, FORBIDDEN, 'Forbidden Resource', detail)