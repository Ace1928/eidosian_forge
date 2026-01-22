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
def test_losingConnectionWithNoAbortTimeOut(self):
    """
        When a L{H2Connection} has timed the connection out but the
        C{abortTimeout} is set to L{None}, the connection is never aborted.
        """
    reactor, conn, transport = self.prepareAbortTest(abortTimeout=None)
    reactor.advance(2 ** 32)
    self.assertTrue(transport.disconnecting)
    self.assertFalse(transport.disconnected)