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
def initiateH2Connection(self, initialData, requestFactory):
    """
        Performs test setup by building a HTTP/2 connection object, a transport
        to back it, a reactor to run in, and sending in some initial data as
        needed.

        @param initialData: The initial HTTP/2 data to be fed into the
            connection after setup.
        @type initialData: L{bytes}

        @param requestFactory: The L{Request} factory to use with the
            connection.
        """
    reactor = task.Clock()
    conn = H2Connection(reactor)
    conn.timeOut = 100
    self.patch_TimeoutMixin_clock(conn, reactor)
    transport = StringTransport()
    conn.requestFactory = _makeRequestProxyFactory(requestFactory)
    conn.makeConnection(transport)
    for byte in iterbytes(initialData):
        conn.dataReceived(byte)
    return (reactor, conn, transport)