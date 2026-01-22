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
def test_tcpKeepAlive(self):
    """
        The transport of a protocol connected with L{IReactorTCP.connectTCP} or
        L{IReactor.TCP.listenTCP} can have its I{SO_KEEPALIVE} state inspected
        and manipulated with L{ITCPTransport.getTcpKeepAlive} and
        L{ITCPTransport.setTcpKeepAlive}.
        """

    def check(serverProtocol, clientProtocol):
        for p in [serverProtocol, clientProtocol]:
            transport = p.transport
            self.assertEqual(transport.getTcpKeepAlive(), 0)
            transport.setTcpKeepAlive(1)
            self.assertEqual(transport.getTcpKeepAlive(), 1)
            transport.setTcpKeepAlive(0)
            self.assertEqual(transport.getTcpKeepAlive(), 0)
    return self._connectedClientAndServerTest(check)