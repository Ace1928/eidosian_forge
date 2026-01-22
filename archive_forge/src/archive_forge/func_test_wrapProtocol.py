import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
def test_wrapProtocol(self):
    """
        L{wrapProtocol}, when passed a L{Protocol} should return something that
        has write(), writeSequence(), loseConnection() methods which call the
        Protocol's dataReceived() and connectionLost() methods, respectively.
        """
    protocol = MockProtocol()
    protocol.transport = StubTransport()
    protocol.connectionMade()
    wrapped = session.wrapProtocol(protocol)
    wrapped.dataReceived(b'dataReceived')
    self.assertEqual(protocol.transport.buf, b'dataReceived')
    wrapped.write(b'data')
    wrapped.writeSequence([b'1', b'2'])
    wrapped.loseConnection()
    self.assertEqual(protocol.data, b'data12')
    protocol.reason.trap(error.ConnectionDone)