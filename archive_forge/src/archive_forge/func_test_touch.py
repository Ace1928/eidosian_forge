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
def test_touch(self):
    """
        L{server.Session.touch} updates L{server.Session.lastModified} and
        delays session timeout.
        """
    self.clock.advance(3)
    self.session.touch()
    self.assertEqual(self.session.lastModified, 3)
    self.session.startCheckingExpiration()
    self.clock.advance(self.session.sessionTimeout - 1)
    self.session.touch()
    self.clock.advance(self.session.sessionTimeout - 1)
    self.assertIn(self.uid, self.site.sessions)
    self.clock.advance(1)
    self.assertNotIn(self.uid, self.site.sessions)