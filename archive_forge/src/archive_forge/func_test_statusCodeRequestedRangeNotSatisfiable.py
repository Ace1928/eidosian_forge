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
def test_statusCodeRequestedRangeNotSatisfiable(self):
    """
        If a range is syntactically invalid due to the start being greater than
        the end, the range header is ignored (the request is responded to as if
        it were not present).
        """
    self.request.requestHeaders.addRawHeader(b'range', b'bytes=20-13')
    self.resource.render(self.request)
    self.assertEqual(self.request.responseCode, http.OK)
    self.assertEqual(b''.join(self.request.written), self.payload)
    self.assertEqual(self.request.responseHeaders.getRawHeaders(b'content-length')[0], b'%d' % (len(self.payload),))