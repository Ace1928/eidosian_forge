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
def test_headersSavedOnResponse(self):
    """
        All headers received by L{HTTPParser} are added to
        L{HTTPParser.headers}.
        """
    protocol = HTTPParser()
    protocol.makeConnection(StringTransport())
    protocol.dataReceived(b'HTTP/1.1 200 OK' + self.sep)
    protocol.dataReceived(b'X-Foo: bar' + self.sep)
    protocol.dataReceived(b'X-Foo: baz' + self.sep)
    protocol.dataReceived(self.sep)
    expected = [(b'X-Foo', [b'bar', b'baz'])]
    self.assertEqual(expected, list(protocol.headers.getAllRawHeaders()))