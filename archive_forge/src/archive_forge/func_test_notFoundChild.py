import os
from twisted.internet import defer
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.web.http import NOT_FOUND
from twisted.web.script import PythonScript, ResourceScriptDirectory
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyRequest
from twisted.web.resource import Resource
def test_notFoundChild(self) -> defer.Deferred[None]:
    """
        L{ResourceScriptDirectory.getChild} returns a resource which renders an
        response with the HTTP I{NOT FOUND} status code if the indicated child
        does not exist as an entry in the directory used to initialized the
        L{ResourceScriptDirectory}.
        """
    path = self.mktemp()
    os.makedirs(path)
    resource = ResourceScriptDirectory(path)
    request = DummyRequest([b'foo'])
    child = resource.getChild('foo', request)
    d = _render(child, request)

    def cbRendered(ignored: object) -> None:
        self.assertEqual(request.responseCode, NOT_FOUND)
    return d.addCallback(cbRendered)