import os
from twisted.internet import defer
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.web.http import NOT_FOUND
from twisted.web.script import PythonScript, ResourceScriptDirectory
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyRequest
from twisted.web.resource import Resource
def test_notFoundRender(self) -> defer.Deferred[None]:
    """
        If the source file a L{PythonScript} is initialized with doesn't exist,
        L{PythonScript.render} sets the HTTP response code to I{NOT FOUND}.
        """
    resource = PythonScript(self.mktemp(), None)
    request = DummyRequest([b''])
    d = _render(resource, request)

    def cbRendered(ignored: object) -> None:
        self.assertEqual(request.responseCode, NOT_FOUND)
    return d.addCallback(cbRendered)