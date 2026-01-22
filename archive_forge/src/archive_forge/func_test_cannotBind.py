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
def test_cannotBind(self):
    """
        L{IReactorTCP.listenTCP} raises L{error.CannotListenError} if the
        address to listen on is already in use.
        """
    f = MyServerFactory()
    p1 = reactor.listenTCP(0, f, interface='127.0.0.1')
    self.addCleanup(p1.stopListening)
    n = p1.getHost().port
    dest = p1.getHost()
    self.assertEqual(dest.type, 'TCP')
    self.assertEqual(dest.host, '127.0.0.1')
    self.assertEqual(dest.port, n)
    self.assertRaises(error.CannotListenError, reactor.listenTCP, n, f, interface='127.0.0.1')