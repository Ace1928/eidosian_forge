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
def test_requestExec(self):
    """
        When a client requests a command, the SSHSession object should get
        the command by getting an ISession adapter for the avatar, then
        calling execCommand with a ProcessProtocol to attach and the
        command line.
        """
    ret = self.session.requestReceived(b'exec', common.NS(b'failure'))
    self.assertFalse(ret)
    self.assertRequestRaisedRuntimeError()
    self.assertIsNone(self.session.client)
    self.assertTrue(self.session.requestReceived(b'exec', common.NS(b'success')))
    self.assertSessionIsStubSession()
    self.assertIsInstance(self.session.client, session.SSHSessionProcessProtocol)
    self.assertIs(self.session.session.execProtocol, self.session.client)
    self.assertEqual(self.session.session.execCommandLine, b'success')