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
def testStringToDatetime(self):
    dateStrings = [b'Sun, 06 Nov 1994 08:49:37 GMT', b'06 Nov 1994 08:49:37 GMT', b'Sunday, 06-Nov-94 08:49:37 GMT', b'06-Nov-94 08:49:37 GMT', b'Sunday, 06-Nov-1994 08:49:37 GMT', b'06-Nov-1994 08:49:37 GMT', b'Sun Nov  6 08:49:37 1994', b'Nov  6 08:49:37 1994']
    dateInt = calendar.timegm((1994, 11, 6, 8, 49, 37, 6, 6, 0))
    for dateString in dateStrings:
        self.assertEqual(http.stringToDatetime(dateString), dateInt)
    self.assertEqual(http.stringToDatetime(b'Thursday, 29-Sep-16 17:15:29 GMT'), calendar.timegm((2016, 9, 29, 17, 15, 29, 3, 273, 0)))