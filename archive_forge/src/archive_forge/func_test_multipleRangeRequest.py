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
def test_multipleRangeRequest(self):
    """
        The response to a request for multiple bytes ranges is a MIME-ish
        multipart response.
        """
    startEnds = [(0, 2), (20, 30), (40, 50)]
    rangeHeaderValue = b','.join([networkString(f'{s}-{e}') for s, e in startEnds])
    self.request.requestHeaders.addRawHeader(b'range', b'bytes=' + rangeHeaderValue)
    self.resource.render(self.request)
    self.assertEqual(self.request.responseCode, http.PARTIAL_CONTENT)
    boundary = re.match(b'^multipart/byteranges; boundary="(.*)"$', self.request.responseHeaders.getRawHeaders(b'content-type')[0]).group(1)
    parts = self.parseMultipartBody(b''.join(self.request.written), boundary)
    self.assertEqual(len(startEnds), len(parts))
    for part, (s, e) in zip(parts, startEnds):
        self.assertEqual(networkString(self.resource.type), part[b'contentType'])
        start, end, size = part[b'contentRange']
        self.assertEqual(int(start), s)
        self.assertEqual(int(end), e)
        self.assertEqual(int(size), self.resource.getFileSize())
        self.assertEqual(self.payload[s:e + 1], part[b'body'])