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
def prepareAbortTest(self, abortTimeout=_DEFAULT):
    """
        Does the common setup for tests that want to test the aborting
        functionality of the HTTP/2 stack.

        @param abortTimeout: The value to use for the abortTimeout. Defaults to
            whatever is set on L{H2Connection.abortTimeout}.
        @type abortTimeout: L{int} or L{None}

        @return: A tuple of the reactor being used for the connection, the
            connection itself, and the transport.
        """
    if abortTimeout is self._DEFAULT:
        abortTimeout = H2Connection.abortTimeout
    frameFactory = FrameFactory()
    initialData = frameFactory.clientConnectionPreface()
    reactor, conn, transport = self.initiateH2Connection(initialData, requestFactory=DummyHTTPHandler)
    conn.abortTimeout = abortTimeout
    reactor.advance(100)
    self.assertTimedOut(transport.value(), frameCount=2, errorCode=h2.errors.ErrorCodes.NO_ERROR, lastStreamID=0)
    self.assertTrue(transport.disconnecting)
    self.assertFalse(transport.disconnected)
    return (reactor, conn, transport)