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
def test_bodyDataFinishedBeforeStartProducing(self):
    """
        If the entire body is delivered to the L{Response} before the
        response's C{deliverBody} method is called, the protocol passed to
        C{deliverBody} is immediately given the body data and then
        disconnected.
        """
    transport = StringTransport()
    response = justTransportResponse(transport)
    response._bodyDataReceived(b'foo')
    response._bodyDataReceived(b'bar')
    response._bodyDataFinished()
    protocol = AccumulatingProtocol()
    response.deliverBody(protocol)
    self.assertEqual(protocol.data, b'foobar')
    protocol.closedReason.trap(ResponseDone)