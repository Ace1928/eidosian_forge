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
def test_noResponseAtAll(self):
    """
        If no response at all was received and the connection is lost, the
        resulting error is L{ResponseNeverReceived}.
        """
    protocol = HTTPClientParser(Request(b'HEAD', b'/', _boringHeaders, None), lambda ign: None)
    d = protocol._responseDeferred
    protocol.makeConnection(StringTransport())
    protocol.connectionLost(ConnectionLost())
    return self.assertFailure(d, ResponseNeverReceived)