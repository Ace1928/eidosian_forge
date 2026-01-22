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
def test_simplestSite(self):
    """
        L{Site.getResourceFor} returns the C{b""} child of the root resource it
        is constructed with when processing a request for I{/}.
        """
    sres1 = SimpleResource()
    sres2 = SimpleResource()
    sres1.putChild(b'', sres2)
    site = server.Site(sres1)
    self.assertIdentical(site.getResourceFor(DummyRequest([b''])), sres2, 'Got the wrong resource.')