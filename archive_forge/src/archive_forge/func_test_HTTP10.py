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
def test_HTTP10(self):
    """
        HTTP/1.0 requests do not get 100-continue returned, even if 'Expect:
        100-continue' is included (RFC 2616 10.1.1).
        """
    transport = StringTransport()
    channel = http.HTTPChannel()
    channel.requestFactory = DummyHTTPHandlerProxy
    channel.makeConnection(transport)
    channel.dataReceived(b'GET / HTTP/1.0\r\n')
    channel.dataReceived(b'Host: www.example.com\r\n')
    channel.dataReceived(b'Content-Length: 3\r\n')
    channel.dataReceived(b'Expect: 100-continue\r\n')
    channel.dataReceived(b'\r\n')
    self.assertEqual(transport.value(), b'')
    channel.dataReceived(b'abc')
    self.assertResponseEquals(transport.value(), [(b'HTTP/1.0 200 OK', b'Command: GET', b'Content-Length: 13', b'Version: HTTP/1.0', b'Request: /', b"'''\n3\nabc'''\n")])