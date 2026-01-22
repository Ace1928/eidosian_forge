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
def test_notAllowedQuoting(self):
    """
        When an unsupported method response is generated, an HTML message will
        be displayed.  That message should include a quoted form of the URI and,
        since that value come from a browser and shouldn't necessarily be
        trusted.
        """
    req = self._getReq()
    req.requestReceived(b'POST', b'/gettableresource?value=<script>bad', b'HTTP/1.0')
    self.assertEqual(req.code, 405)
    renderedPage = req.transport.written.getvalue()
    self.assertNotIn(b'<script>bad', renderedPage)
    self.assertIn(b'&lt;script&gt;bad', renderedPage)