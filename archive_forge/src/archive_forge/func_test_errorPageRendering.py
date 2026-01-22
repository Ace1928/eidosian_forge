from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def test_errorPageRendering(self) -> None:
    """
        L{ErrorPage.render} returns a C{bytes} describing the error defined by
        the response code and message passed to L{ErrorPage.__init__}.  It also
        uses that response code to set the response code on the L{Request}
        passed in.
        """
    code = 321
    brief = 'brief description text'
    detail = 'much longer text might go here'
    page = self.errorPage(code, brief, detail)
    self._pageRenderingTest(page, code, brief, detail)