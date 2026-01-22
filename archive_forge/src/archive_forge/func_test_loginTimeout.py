from types import ModuleType
from typing import Optional
from zope.interface import implementer
from twisted.conch.error import ConchError, ValidPublicKey
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IAnonymous, ISSHPrivateKey, IUsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, task
from twisted.protocols import loopback
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_loginTimeout(self):
    """
        Test that the login times out.
        """
    timeoutAuthServer = userauth.SSHUserAuthServer()
    timeoutAuthServer.clock = task.Clock()
    timeoutAuthServer.transport = FakeTransport(self.portal)
    timeoutAuthServer.serviceStarted()
    timeoutAuthServer.clock.advance(11 * 60 * 60)
    timeoutAuthServer.serviceStopped()
    self.assertEqual(timeoutAuthServer.transport.packets, [(transport.MSG_DISCONNECT, b'\x00' * 3 + bytes((transport.DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE,)) + NS(b'you took too long') + NS(b''))])
    self.assertTrue(timeoutAuthServer.transport.lostConnection)