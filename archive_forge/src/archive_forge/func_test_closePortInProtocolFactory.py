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
def test_closePortInProtocolFactory(self):
    """
        A port created with L{IReactorTCP.listenTCP} can be connected to with
        L{IReactorTCP.connectTCP}.
        """
    f = ClosingFactory()
    port = reactor.listenTCP(0, f, interface='127.0.0.1')
    f.port = port
    self.addCleanup(f.cleanUp)
    portNumber = port.getHost().port
    clientF = MyClientFactory()
    reactor.connectTCP('127.0.0.1', portNumber, clientF)

    def check(x):
        self.assertTrue(clientF.protocol.made)
        self.assertTrue(port.disconnected)
        clientF.lostReason.trap(error.ConnectionDone)
    return clientF.deferred.addCallback(check)