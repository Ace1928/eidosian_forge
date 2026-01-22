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
def test_addCookieWithAllArgumentsUnicode(self):
    """
        L{http.Request.addCookie} adds a new cookie to be sent with the
        response. L{unicode} arguments are encoded using UTF-8.
        """
    expectedCookieValue = b'foo=bar; Expires=Fri, 31 Dec 9999 23:59:59 GMT; Domain=.example.com; Path=/; Max-Age=31536000; Comment=test; Secure; HttpOnly'
    self._checkCookie(expectedCookieValue, 'foo', 'bar', expires='Fri, 31 Dec 9999 23:59:59 GMT', domain='.example.com', path='/', max_age='31536000', comment='test', secure=True, httpOnly=True)