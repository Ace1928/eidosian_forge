import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import (
def test_notifyOnResetStream(self):
    """
        A HTTP/2 reset stream fires the L{http.Request.notifyFinish} deferred
        with L{ConnectionLost}.
        """
    connection = H2Connection()
    connection.requestFactory = NotifyingRequestFactory(DelayedHTTPHandler)
    frameFactory, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
    deferreds = connection.requestFactory.results
    self.assertEqual(len(deferreds), 1)

    def callback(ign):
        self.fail("Didn't errback, called back instead")

    def errback(reason):
        self.assertIsInstance(reason, failure.Failure)
        self.assertIs(reason.type, error.ConnectionLost)
        return None
    d = deferreds[0]
    d.addCallbacks(callback, errback)
    invalidData = frameFactory.buildRstStreamFrame(streamID=1).serialize()
    connection.dataReceived(invalidData)
    return d