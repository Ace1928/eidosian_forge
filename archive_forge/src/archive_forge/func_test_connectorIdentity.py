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
def test_connectorIdentity(self):
    """
        L{IReactorTCP.connectTCP} returns an object which provides
        L{IConnector}.  The destination of the connector is the address which
        was passed to C{connectTCP}.  The same connector object is passed to
        the factory's C{startedConnecting} method as to the factory's
        C{clientConnectionLost} method.
        """
    serverFactory = ClosingFactory()
    reactor = self.buildReactor()
    tcpPort = reactor.listenTCP(0, serverFactory, interface=self.interface)
    serverFactory.port = tcpPort
    portNumber = tcpPort.getHost().port
    seenConnectors = []
    seenFailures = []
    clientFactory = ClientStartStopFactory()
    clientFactory.clientConnectionLost = lambda connector, reason: (seenConnectors.append(connector), seenFailures.append(reason))
    clientFactory.startedConnecting = seenConnectors.append
    connector = reactor.connectTCP(self.interface, portNumber, clientFactory)
    self.assertTrue(IConnector.providedBy(connector))
    dest = connector.getDestination()
    self.assertEqual(dest.type, 'TCP')
    self.assertEqual(dest.host, self.interface)
    self.assertEqual(dest.port, portNumber)
    clientFactory.whenStopped.addBoth(lambda _: reactor.stop())
    self.runReactor(reactor)
    seenFailures[0].trap(ConnectionDone)
    self.assertEqual(seenConnectors, [connector, connector])