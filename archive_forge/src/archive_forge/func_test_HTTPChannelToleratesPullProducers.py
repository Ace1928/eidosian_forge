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
def test_HTTPChannelToleratesPullProducers(self):
    """
        If the L{HTTPChannel} has a L{IPullProducer} registered with it it can
        adapt that producer into an L{IPushProducer}.
        """
    channel, transport = self.buildChannelAndTransport(StringTransport(), DummyPullProducerHandler)
    transport = StringTransport()
    channel = http.HTTPChannel()
    channel.requestFactory = DummyPullProducerHandlerProxy
    channel.makeConnection(transport)
    channel.dataReceived(self.request)
    request = channel.requests[0].original
    responseComplete = request._actualProducer.result

    def validate(ign):
        responseBody = transport.value().split(b'\r\n\r\n', 1)[1]
        expectedResponseBody = b'1\r\n0\r\n1\r\n1\r\n1\r\n2\r\n1\r\n3\r\n1\r\n4\r\n1\r\n5\r\n1\r\n6\r\n1\r\n7\r\n1\r\n8\r\n1\r\n9\r\n'
        self.assertEqual(responseBody, expectedResponseBody)
    return responseComplete.addCallback(validate)