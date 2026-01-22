import base64
import calendar
import random
from io import BytesIO
from itertools import cycle
from typing import Sequence, Union
from unittest import skipIf
from urllib.parse import clear_cache  # type: ignore[attr-defined]
from urllib.parse import urlparse, urlunsplit
from zope.interface import directlyProvides, providedBy, provider
from zope.interface.verify import verifyObject
import hamcrest
from twisted.internet import address
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols import loopback
from twisted.python.compat import iterbytes, networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
from twisted.web import http, http_headers, iweb
from twisted.web.http import PotentialDataLoss, _DataLoss, _IdentityTransferDecoder
from twisted.web.test.requesthelper import (
from ._util import assertIsFilesystemTemporary
def test_firstWriteLastModified(self):
    """
        For an HTTP 1.0 request for a resource with a known last modified time,
        L{http.Request.write} sends an HTTP Response-Line, whatever response
        headers are set, and a last-modified header with that time.
        """
    channel = DummyChannel()
    req = http.Request(channel, False)
    trans = StringTransport()
    channel.transport = trans
    req.setResponseCode(200)
    req.clientproto = b'HTTP/1.0'
    req.lastModified = 0
    req.responseHeaders.setRawHeaders(b'test', [b'lemur'])
    req.write(b'Hello')
    self.assertResponseEquals(trans.value(), [(b'HTTP/1.0 200 OK', b'Test: lemur', b'Last-Modified: Thu, 01 Jan 1970 00:00:00 GMT', b'Hello')])