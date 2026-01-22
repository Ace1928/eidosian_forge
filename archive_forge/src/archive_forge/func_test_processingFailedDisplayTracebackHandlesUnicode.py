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
def test_processingFailedDisplayTracebackHandlesUnicode(self):
    """
        L{Request.processingFailed} when the site has C{displayTracebacks} set
        to C{True} writes out the failure, making UTF-8 items into HTML
        entities.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    d = DummyChannel()
    request = server.Request(d, 1)
    request.site = server.Site(resource.Resource())
    request.site.displayTracebacks = True
    fail = failure.Failure(Exception('☃'))
    request.processingFailed(fail)
    self.assertIn(b'&#9731;', request.transport.written.getvalue())
    self.flushLoggedErrors(UnicodeError)
    event = logObserver[0]
    f = event['log_failure']
    self.assertIsInstance(f.value, Exception)
    self.assertEqual(1, len(self.flushLoggedErrors()))