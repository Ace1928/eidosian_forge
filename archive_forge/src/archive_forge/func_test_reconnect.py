import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
def test_reconnect(self):
    """
        Calling L{IConnector.connect} in C{Factory.clientConnectionLost} causes
        a new connection attempt to be made.
        """
    serverFactory = ClosingFactory()
    reactor = self.buildReactor()
    tcpPort = reactor.listenTCP(0, serverFactory, interface=self.interface)
    serverFactory.port = tcpPort
    portNumber = tcpPort.getHost().port
    clientFactory = MyClientFactory()

    def clientConnectionLost(connector, reason):
        connector.connect()
    clientFactory.clientConnectionLost = clientConnectionLost
    reactor.connectTCP(self.interface, portNumber, clientFactory)
    protocolMadeAndClosed = []

    def reconnectFailed(ignored):
        p = clientFactory.protocol
        protocolMadeAndClosed.append((p.made, p.closed))
        reactor.stop()
    clientFactory.failDeferred.addCallback(reconnectFailed)
    self.runReactor(reactor)
    clientFactory.reason.trap(ConnectionRefusedError)
    self.assertEqual(protocolMadeAndClosed, [(1, 1)])