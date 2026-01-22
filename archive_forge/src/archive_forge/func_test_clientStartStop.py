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
def test_clientStartStop(self):
    """
        The factory passed to L{IReactorTCP.connectTCP} should be started when
        the connection attempt starts and stopped when it is over.
        """
    f = ClosingFactory()
    p = reactor.listenTCP(0, f, interface='127.0.0.1')
    f.port = p
    self.addCleanup(f.cleanUp)
    portNumber = p.getHost().port
    factory = ClientStartStopFactory()
    reactor.connectTCP('127.0.0.1', portNumber, factory)
    self.assertTrue(factory.started)
    return loopUntil(lambda: factory.stopped)