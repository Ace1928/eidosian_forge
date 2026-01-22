from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
def test_connectionControlHeaders(self):
    """
        L{HTTPParser.isConnectionControlHeader} returns C{True} for headers
        which are always connection control headers (similar to "hop-by-hop"
        headers from RFC 2616 section 13.5.1) and C{False} for other headers.
        """
    protocol = HTTPParser()
    connHeaderNames = [b'content-length', b'connection', b'keep-alive', b'te', b'trailers', b'transfer-encoding', b'upgrade', b'proxy-connection']
    for header in connHeaderNames:
        self.assertTrue(protocol.isConnectionControlHeader(header), "Expecting %r to be a connection control header, but wasn't" % (header,))
    self.assertFalse(protocol.isConnectionControlHeader(b'date'), "Expecting the arbitrarily selected 'date' header to not be a connection control header, but was.")