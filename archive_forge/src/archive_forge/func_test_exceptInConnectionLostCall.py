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
def test_exceptInConnectionLostCall(self):
    """
        If C{connectionLost} is called directory on a port object and that the
        server factory raises an exception in C{stopFactory}, the exception is
        passed through to the caller.

        C{connectionLost} is called by L{reactor.disconnectAll} at shutdown.
        """
    serverFactory = MyServerFactory()

    def raiseException():
        raise RuntimeError('An error')
    serverFactory.stopFactory = raiseException
    port = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
    self.assertRaises(RuntimeError, port.connectionLost, None)