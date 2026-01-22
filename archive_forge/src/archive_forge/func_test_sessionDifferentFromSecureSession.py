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
def test_sessionDifferentFromSecureSession(self):
    """
        L{Request.session} and L{Request.secure_session} should be two separate
        sessions with unique ids and different cookies.
        """
    d = DummyChannel()
    d.transport = DummyChannel.SSL()
    request = server.Request(d, 1)
    request.site = server.Site(resource.Resource())
    request.sitepath = []
    secureSession = request.getSession()
    self.assertIsNotNone(secureSession)
    self.addCleanup(secureSession.expire)
    self.assertEqual(request.cookies[0].split(b'=')[0], b'TWISTED_SECURE_SESSION')
    session = request.getSession(forceNotSecure=True)
    self.assertIsNotNone(session)
    self.assertEqual(request.cookies[1].split(b'=')[0], b'TWISTED_SESSION')
    self.addCleanup(session.expire)
    self.assertNotEqual(session.uid, secureSession.uid)