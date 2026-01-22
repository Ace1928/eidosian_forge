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
def test_requestShellWithData(self):
    """
        When a client executes a shell, it should be able to give pass data
        back and forth between the local and the remote side.
        """
    ret = self.session.requestReceived(b'shell', b'')
    self.assertTrue(ret)
    self.assertSessionIsStubSession()
    self.session.dataReceived(b'some data\x00')
    self.assertEqual(self.session.session.shellTransport.data, b'some data\x00')
    self.assertEqual(self.session.conn.data[self.session], [b'some data\x00', b'\r\n'])
    self.assertTrue(self.session.session.shellTransport.closed)
    self.assertEqual(self.session.conn.requests[self.session], [(b'exit-status', b'\x00\x00\x00\x00', False)])