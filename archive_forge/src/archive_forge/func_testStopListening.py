import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def testStopListening(self):
    """
        The L{IListeningPort} returned by L{IReactorTCP.listenTCP} can be
        stopped with its C{stopListening} method.  After the L{Deferred} it
        (optionally) returns has been called back, the port number can be bound
        to a new server.
        """
    f = MyServerFactory()
    port = reactor.listenTCP(0, f, interface='127.0.0.1')
    n = port.getHost().port

    def cbStopListening(ignored):
        port = reactor.listenTCP(n, f, interface='127.0.0.1')
        return port.stopListening()
    d = defer.maybeDeferred(port.stopListening)
    d.addCallback(cbStopListening)
    return d