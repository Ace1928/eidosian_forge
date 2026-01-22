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
def test_staticFile(self):
    """
        If a request is made which encounters a L{File} before a final segment
        which names a file in the path the L{File} was created with, that file
        is served as the response to the request.
        """
    base = FilePath(self.mktemp())
    base.makedirs()
    base.child('foo.bar').setContent(b'baz')
    file = static.File(base.path)
    request = DummyRequest([b'foo.bar'])
    child = resource.getChildForRequest(file, request)
    d = self._render(child, request)

    def cbRendered(ignored):
        self.assertEqual(b''.join(request.written), b'baz')
        self.assertEqual(request.responseHeaders.getRawHeaders(b'content-length')[0], b'3')
    d.addCallback(cbRendered)
    return d