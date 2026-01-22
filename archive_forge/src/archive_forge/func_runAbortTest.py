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
def runAbortTest(self, clientClass, serverClass, clientConnectionLostReason=None):
    """
        A test runner utility function, which hooks up a matched pair of client
        and server protocols.

        We then run the reactor until both sides have disconnected, and then
        verify that the right exception resulted.
        """
    clientExpectedExceptions = (ConnectionAborted, ConnectionLost)
    serverExpectedExceptions = (ConnectionLost, ConnectionDone)
    if useSSL:
        clientExpectedExceptions = clientExpectedExceptions + (SSL.Error,)
        serverExpectedExceptions = serverExpectedExceptions + (SSL.Error,)
    client = clientClass()
    server = serverClass()
    client.otherProtocol = server
    server.otherProtocol = client
    reactor = runProtocolsWithReactor(self, server, client, self.endpoints)
    self.assertEqual(reactor.removeAll(), [])
    self.assertEqual(reactor.getDelayedCalls(), [])
    if clientConnectionLostReason is not None:
        self.assertIsInstance(client.disconnectReason.value, (clientConnectionLostReason,) + clientExpectedExceptions)
    else:
        self.assertIsInstance(client.disconnectReason.value, clientExpectedExceptions)
    self.assertIsInstance(server.disconnectReason.value, serverExpectedExceptions)