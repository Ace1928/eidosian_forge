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
def test_tooLongTrailerHeader(self):
    """
        L{_ChunkedTransferDecoder.dataReceived} raises
        L{_MalformedChunkedDataError} when the trailing headers data is too long.
        """
    p = http._ChunkedTransferDecoder(lambda b: None, lambda b: None)
    p._maxTrailerHeadersSize = 10
    self.assertRaises(http._MalformedChunkedDataError, p.dataReceived, b'3\r\nabc\r\n0\r\nTotal-Trailer: header;greater-then=10\r\n\r\n')