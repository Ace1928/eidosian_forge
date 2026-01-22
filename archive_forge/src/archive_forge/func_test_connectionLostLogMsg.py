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
def test_connectionLostLogMsg(self):
    """
        When a connection is lost, an informative message should be logged
        (see L{getExpectedConnectionLostLogMsg}): an address identifying
        the port and the fact that it was closed.
        """
    loggedMessages = []

    def logConnectionLostMsg(eventDict):
        loggedMessages.append(log.textFromEventDict(eventDict))
    reactor = self.buildReactor()
    p = self.getListeningPort(reactor, ServerFactory())
    expectedMessage = self.getExpectedConnectionLostLogMsg(p)
    log.addObserver(logConnectionLostMsg)

    def stopReactor(ignored):
        log.removeObserver(logConnectionLostMsg)
        reactor.stop()

    def doStopListening():
        log.addObserver(logConnectionLostMsg)
        maybeDeferred(p.stopListening).addCallback(stopReactor)
    reactor.callWhenRunning(doStopListening)
    reactor.run()
    self.assertIn(expectedMessage, loggedMessages)