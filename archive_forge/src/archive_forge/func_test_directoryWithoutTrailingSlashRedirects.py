import errno
import inspect
import mimetypes
import os
import re
import sys
import warnings
from io import BytesIO as StringIO
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import abstract, interfaces
from twisted.python import compat, log
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.web import http, resource, script, static
from twisted.web._responses import FOUND
from twisted.web.server import UnsupportedMethod
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyRequest
def test_directoryWithoutTrailingSlashRedirects(self):
    """
        A request for a path which is a directory but does not have a trailing
        slash will be redirected to a URL which does have a slash by L{File}.
        """
    base = FilePath(self.mktemp())
    base.makedirs()
    base.child('folder').makedirs()
    file = static.File(base.path)
    request = DummyRequest([b'folder'])
    request.uri = b'http://dummy/folder#baz?foo=bar'
    child = resource.getChildForRequest(file, request)
    self.successResultOf(self._render(child, request))
    self.assertEqual(request.responseCode, FOUND)
    self.assertEqual(request.responseHeaders.getRawHeaders(b'location'), [b'http://dummy/folder/#baz?foo=bar'])