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
def test_implicitStart(self):
    """
        If the start byte position is omitted but the end byte position is
        supplied, then the range is treated as requesting the last -N bytes of
        the resource, where N is the end byte position.
        """
    self.request.requestHeaders.addRawHeader(b'range', b'bytes=-17')
    self.resource.render(self.request)
    self.assertEqual(b''.join(self.request.written), self.payload[-17:])
    self.assertEqual(len(b''.join(self.request.written)), 17)
    self.assertEqual(self.request.responseCode, http.PARTIAL_CONTENT)
    self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-range')[0], b'bytes 47-63/64')
    self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-length')[0], b'17')