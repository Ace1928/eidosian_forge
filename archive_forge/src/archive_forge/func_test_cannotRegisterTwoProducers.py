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
def test_cannotRegisterTwoProducers(self):
    """
        The L{H2Stream} object forbids registering two producers.
        """
    connection = H2Connection()
    connection.requestFactory = DummyProducerHandlerProxy
    self.connectAndReceive(connection, self.getRequestHeaders, [])
    stream = connection.streams[1]
    request = stream._request.original
    self.assertRaises(ValueError, stream.registerProducer, request, True)