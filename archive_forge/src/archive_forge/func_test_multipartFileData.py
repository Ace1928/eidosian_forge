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
def test_multipartFileData(self):
    """
        If the request has a Content-Type of C{multipart/form-data},
        and the form data is parseable and contains files, the file
        portions will be added to the request's args.
        """
    processed = []

    class MyRequest(http.Request):

        def process(self):
            processed.append(self)
            self.write(b'done')
            self.finish()
    body = b'-----------------------------738837029596785559389649595\nContent-Disposition: form-data; name="uploadedfile"; filename="test"\nContent-Type: application/octet-stream\n\nabasdfg\n-----------------------------738837029596785559389649595--\n'
    req = 'POST / HTTP/1.0\nContent-Type: multipart/form-data; boundary=---------------------------738837029596785559389649595\nContent-Length: ' + str(len(body.replace(b'\n', b'\r\n'))) + '\n\n\n'
    channel = self.runRequest(req.encode('ascii') + body, MyRequest, success=False)
    self.assertEqual(channel.transport.value(), b'HTTP/1.0 200 OK\r\n\r\ndone')
    self.assertEqual(len(processed), 1)
    self.assertEqual(processed[0].args, {b'uploadedfile': [b'abasdfg']})