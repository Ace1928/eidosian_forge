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
def test_setEnvMultiplexShare(self):
    """
        Calling another session service after setenv will provide the
        previous session with the environment variables.
        """
    test_session = self.getSSHSession()
    test_session.requestReceived(b'env', common.NS(b'Key1') + common.NS(b'Value 1'))
    test_session.requestReceived(b'env', common.NS(b'Key2') + common.NS(b'Value2'))
    test_session.requestReceived(b'pty_req', session.packRequest_pty_req(b'term', (0, 0, 0, 0), b''))
    self.assertIsInstance(test_session.session, StubSessionForStubAvatarWithEnv)
    self.assertEqual({b'Key1': b'Value 1', b'Key2': b'Value2'}, test_session.session.environAtPty)