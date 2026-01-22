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
def test_tooManyAttempts(self):
    """
        Test that the server disconnects if the client fails authentication
        too many times.
        """
    packet = b''.join([NS(b'foo'), NS(b'none'), NS(b'password'), b'\x00', NS(b'bar')])
    self.authServer.clock = task.Clock()
    for i in range(21):
        d = self.authServer.ssh_USERAUTH_REQUEST(packet)
        self.authServer.clock.advance(2)

    def check(ignored):
        self.assertEqual(self.authServer.transport.packets[-1], (transport.MSG_DISCONNECT, b'\x00' * 3 + bytes((transport.DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE,)) + NS(b'too many bad auths') + NS(b'')))
    return d.addCallback(check)