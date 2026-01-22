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
def test_malformedChunkedEncoding(self):
    """
        If a request uses the I{chunked} transfer encoding, but provides an
        invalid chunk length value, the request fails with a 400 error.
        """
    httpRequest = b"GET / HTTP/1.1\nContent-Type: text/plain\nTransfer-Encoding: chunked\n\nMALFORMED_LINE_THIS_SHOULD_BE_'6'\nHello,\n14\n spam,eggs spam spam\n0\n\n"
    didRequest = []

    class MyRequest(http.Request):

        def process(self):
            didRequest.append(True)
    channel = self.runRequest(httpRequest, MyRequest, success=False)
    self.assertFalse(didRequest, 'Request.process called')
    self.assertEqual(channel.transport.value(), b'HTTP/1.1 400 Bad Request\r\n\r\n')
    self.assertTrue(channel.transport.disconnecting)