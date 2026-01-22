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
def oneTransportTest(testMethod):
    """
    Decorate a L{ReactorBuilder} test function which tests one reactor and one
    connected transport.  Run that test method in the context of
    C{connectionMade}, and immediately drop the connection (and end the test)
    when that completes.

    @param testMethod: A unit test method on a L{ReactorBuilder} test suite;
        taking two additional parameters; a C{reactor} as built by the
        L{ReactorBuilder}, and an L{ITCPTransport} provider.
    @type testMethod: 3-argument C{function}

    @return: a no-argument test method.
    @rtype: 1-argument C{function}
    """

    @wraps(testMethod)
    def actualTestMethod(builder):
        other = ConnectableProtocol()

        class ServerProtocol(ConnectableProtocol):

            def connectionMade(self):
                try:
                    testMethod(builder, self.reactor, self.transport)
                finally:
                    if self.transport is not None:
                        self.transport.loseConnection()
                    if other.transport is not None:
                        other.transport.loseConnection()
        serverProtocol = ServerProtocol()
        runProtocolsWithReactor(builder, serverProtocol, other, TCPCreator())
    return actualTestMethod