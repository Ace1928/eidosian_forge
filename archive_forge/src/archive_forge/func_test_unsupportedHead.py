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
def test_unsupportedHead(self):
    """
        HEAD requests against resource that only claim support for GET
        should not include a body in the response.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    resource = HeadlessResource()
    req = self._getReq(resource)
    req.requestReceived(b'HEAD', b'/newrender', b'HTTP/1.0')
    headers, body = req.transport.written.getvalue().split(b'\r\n\r\n')
    self.assertEqual(req.code, 200)
    self.assertEqual(body, b'')
    self.assertEquals(2, len(logObserver))