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
def test_sendChunkedRequestBodyFinishedWithErrorThenWriteMore(self):
    """
        If the request body producer with an unknown length tries to write
        after firing the L{Deferred} returned by its C{startProducing} method
        with a L{Failure}, the C{write} call raises an exception and does not
        write anything to the underlying transport.
        """
    d = self.test_sendChunkedRequestBodyFinishedThenWriteMore(Failure(ArbitraryException()))
    return self.assertFailure(d, ArbitraryException)