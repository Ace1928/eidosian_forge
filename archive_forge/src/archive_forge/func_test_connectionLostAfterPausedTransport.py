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
def test_connectionLostAfterPausedTransport(self):
    """
        Alice connects to Bob.  Alice writes some bytes and then shuts down the
        connection.  Bob receives the bytes from the connection and then pauses
        the transport object.  Shortly afterwards Bob resumes the transport
        object.  At that point, Bob is notified that the connection has been
        closed.

        This is no problem for most reactors.  The underlying event notification
        API will probably just remind them that the connection has been closed.
        It is a little tricky for win32eventreactor (MsgWaitForMultipleObjects).
        MsgWaitForMultipleObjects will only deliver the close notification once.
        The reactor needs to remember that notification until Bob resumes the
        transport.
        """

    class Pauser(ConnectableProtocol):

        def __init__(self):
            self.events = []

        def dataReceived(self, bytes):
            self.events.append('paused')
            self.transport.pauseProducing()
            self.reactor.callLater(0, self.resume)

        def resume(self):
            self.events.append('resumed')
            self.transport.resumeProducing()

        def connectionLost(self, reason):
            self.events.append('lost')
            ConnectableProtocol.connectionLost(self, reason)

    class Client(ConnectableProtocol):

        def connectionMade(self):
            self.transport.write(b'some bytes for you')
            self.transport.loseConnection()
    pauser = Pauser()
    runProtocolsWithReactor(self, pauser, Client(), TCPCreator())
    self.assertEqual(pauser.events, ['paused', 'resumed', 'lost'])