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
def test_sendChunkedRequestBodyWithError(self):
    """
        If L{Request} is created with a C{bodyProducer} without a known length
        and the L{Deferred} returned from its C{startProducing} method fires
        with a L{Failure}, the L{Deferred} returned by L{Request.writeTo} fires
        with that L{Failure} and the body producer is unregistered from the
        transport.  The final zero-length chunk is not written to the
        transport.
        """
    producer = StringProducer(UNKNOWN_LENGTH)
    request = Request(b'POST', b'/bar', _boringHeaders, producer)
    writeDeferred = request.writeTo(self.transport)
    self.transport.clear()
    producer.finished.errback(ArbitraryException())

    def cbFailed(ignored):
        self.assertEqual(self.transport.value(), b'')
        self.assertIdentical(self.transport.producer, None)
    d = self.assertFailure(writeDeferred, ArbitraryException)
    d.addCallback(cbFailed)
    return d