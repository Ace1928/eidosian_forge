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
def test_disconnectIfNoMoreAuthentication(self):
    """
        If there are no more available user authentication messages,
        the SSHUserAuthClient should disconnect with code
        DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE.
        """
    self.authClient.ssh_USERAUTH_FAILURE(NS(b'password') + b'\x00')
    self.authClient.ssh_USERAUTH_FAILURE(NS(b'password') + b'\xff')
    self.assertEqual(self.authClient.transport.packets[-1], (transport.MSG_DISCONNECT, b'\x00\x00\x00\x0e' + NS(b'no more authentication methods available') + b'\x00\x00\x00\x00'))