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
def test_doubleHalfClose(self):
    """
        If one side half-closes its connection, and then the other side of the
        connection calls C{loseWriteConnection}, and then C{loseConnection} in
        {writeConnectionLost}, the connection is closed correctly.

        This rather obscure case used to fail (see ticket #3037).
        """

    @implementer(IHalfCloseableProtocol)
    class ListenerProtocol(ConnectableProtocol):

        def readConnectionLost(self):
            self.transport.loseWriteConnection()

        def writeConnectionLost(self):
            self.transport.loseConnection()

    class Client(ConnectableProtocol):

        def connectionMade(self):
            self.transport.loseConnection()
    runProtocolsWithReactor(self, ListenerProtocol(), Client(), TCPCreator())