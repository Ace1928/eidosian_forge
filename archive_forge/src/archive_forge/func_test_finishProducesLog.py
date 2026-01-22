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
def test_finishProducesLog(self):
    """
        L{http.Request.finish} will call the channel's factory to produce a log
        message.
        """
    factory = http.HTTPFactory()
    factory.timeOut = None
    factory._logDateTime = 'sometime'
    factory._logDateTimeCall = True
    factory.startFactory()
    factory.logFile = BytesIO()
    proto = factory.buildProtocol(None)
    val = [b'GET /path HTTP/1.1\r\n', b'\r\n\r\n']
    trans = StringTransport()
    proto.makeConnection(trans)
    for x in val:
        proto.dataReceived(x)
    proto._channel.requests[0].finish()
    self.assertIn(b'sometime "GET /path HTTP/1.1"', factory.logFile.getvalue())