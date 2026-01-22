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
def test_headerStripWhitespace(self):
    """
        Leading and trailing space and tab characters are stripped from
        headers. Other forms of whitespace are preserved.

        See RFC 7230 section 3.2.3 and 3.2.4.
        """
    processed = []

    class MyRequest(http.Request):

        def process(self):
            processed.append(self)
            self.finish()
    requestLines = [b'GET / HTTP/1.0', b'spaces:   spaces were stripped   ', b'tabs: \t\ttabs were stripped\t\t', b'spaces-and-tabs: \t \t spaces and tabs were stripped\t \t', b'line-tab:   \x0b vertical tab was preserved\x0b\t', b'form-feed: \x0c form feed was preserved \x0c  ', b'', b'']
    self.runRequest(b'\n'.join(requestLines), MyRequest, 0)
    [request] = processed
    self.assertEqual(request.requestHeaders.getRawHeaders(b'spaces'), [b'spaces were stripped'])
    self.assertEqual(request.requestHeaders.getRawHeaders(b'tabs'), [b'tabs were stripped'])
    self.assertEqual(request.requestHeaders.getRawHeaders(b'spaces-and-tabs'), [b'spaces and tabs were stripped'])
    self.assertEqual(request.requestHeaders.getRawHeaders(b'line-tab'), [b'\x0b vertical tab was preserved\x0b'])
    self.assertEqual(request.requestHeaders.getRawHeaders(b'form-feed'), [b'\x0c form feed was preserved \x0c'])