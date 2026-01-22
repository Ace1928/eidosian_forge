import os
import zlib
from io import BytesIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import interfaces
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.task import Clock
from twisted.internet.testing import EventLoggingObserver, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python import failure, reflect
from twisted.python.compat import iterbytes
from twisted.python.filepath import FilePath
from twisted.trial import unittest
from twisted.web import error, http, iweb, resource, server
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Request, Site
from twisted.web.static import Data
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from ._util import assertIsFilesystemTemporary
def test_unmodifiedWithContentType(self):
    """
        Similar to L{test_etagMatched}, but the response should include a
        I{Content-Type} header if the application explicitly sets one.

        This I{Content-Type} header SHOULD NOT be present according to RFC 2616,
        section 10.3.5.  It will only be present if the application explicitly
        sets it.
        """
    for line in [b'GET /with-content-type HTTP/1.1', b'If-None-Match: MatchingTag', b'']:
        self.channel.dataReceived(line + b'\r\n')
    result = self.transport.getvalue()
    self.assertEqual(httpCode(result), http.NOT_MODIFIED)
    self.assertEqual(httpBody(result), b'')
    self.assertEqual(httpHeader(result, b'Content-Type'), b'image/jpeg')