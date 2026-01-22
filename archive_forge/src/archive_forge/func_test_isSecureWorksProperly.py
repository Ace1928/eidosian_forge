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
def test_isSecureWorksProperly(self):
    """
        L{Request} objects can correctly ask isSecure on HTTP/2.
        """
    connection = H2Connection()
    connection.requestFactory = DelayedHTTPHandlerProxy
    self.connectAndReceive(connection, self.getRequestHeaders, [])
    request = connection.streams[1]._request.original
    self.assertFalse(request.isSecure())
    connection.streams[1].abortConnection()