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
def test_basicAuthException(self):
    """
        A L{Request} that throws an exception processing basic authorization
        logs an error and uses an empty username and password.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
    requests = []

    class Request(http.Request):

        def process(self):
            self.credentials = (self.getUser(), self.getPassword())
            requests.append(self)
    u = b'foo'
    p = b'bar'
    s = base64.b64encode(b':'.join((u, p)))
    f = b'GET / HTTP/1.0\nAuthorization: Basic ' + s + b'\n\n'
    self.patch(base64, 'b64decode', lambda x: [])
    self.runRequest(f, Request, 0)
    req = requests.pop()
    self.assertEqual((b'', b''), req.credentials)
    self.assertEquals(1, len(logObserver))
    event = logObserver[0]
    f = event['log_failure']
    self.assertIsInstance(f.value, AttributeError)
    self.flushLoggedErrors(AttributeError)