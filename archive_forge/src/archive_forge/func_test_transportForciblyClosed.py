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
def test_transportForciblyClosed(self):
    """
        If a timed out transport doesn't close after 15 seconds, the
        L{HTTPChannel} will forcibly close it.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    clock = Clock()
    transport = StringTransport()
    factory = http.HTTPFactory()
    protocol = factory.buildProtocol(None)
    protocol = parametrizeTimeoutMixin(protocol, clock)
    protocol.makeConnection(transport)
    protocol.dataReceived(b'POST / HTTP/1.0\r\nContent-Length: 2\r\n\r\n')
    self.assertFalse(transport.disconnecting)
    self.assertFalse(transport.disconnected)
    clock.advance(60)
    self.assertTrue(transport.disconnecting)
    self.assertFalse(transport.disconnected)
    self.assertEquals(1, len(logObserver))
    event = logObserver[0]
    self.assertIn('Timing out client: {peer}', event['log_format'])
    clock.advance(14)
    self.assertTrue(transport.disconnecting)
    self.assertFalse(transport.disconnected)
    clock.advance(1)
    self.assertTrue(transport.disconnecting)
    self.assertTrue(transport.disconnected)
    self.assertEquals(2, len(logObserver))
    event = logObserver[1]
    self.assertEquals('Forcibly timing out client: {peer}', event['log_format'])