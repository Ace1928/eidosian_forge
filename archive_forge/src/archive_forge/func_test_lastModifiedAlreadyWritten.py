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
def test_lastModifiedAlreadyWritten(self):
    """
        If the last-modified header already exists in the L{http.Request}
        response headers, the lastModified attribute is ignored and a message
        is logged.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    channel = DummyChannel()
    req = http.Request(channel, False)
    trans = StringTransport()
    channel.transport = trans
    req.setResponseCode(200)
    req.clientproto = b'HTTP/1.0'
    req.lastModified = 1000000000
    req.responseHeaders.setRawHeaders(b'last-modified', [b'Thu, 01 Jan 1970 00:00:00 GMT'])
    req.write(b'Hello')
    self.assertResponseEquals(trans.value(), [(b'HTTP/1.0 200 OK', b'Last-Modified: Thu, 01 Jan 1970 00:00:00 GMT', b'Hello')])
    self.assertEquals(1, len(logObserver))
    event = logObserver[0]
    self.assertEquals('Warning: last-modified specified both in header list and lastModified attribute.', event['log_format'])