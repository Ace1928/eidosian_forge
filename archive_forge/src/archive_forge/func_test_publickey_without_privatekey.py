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
def test_publickey_without_privatekey(self):
    """
        If the SSHUserAuthClient doesn't return anything from signData,
        the client should start the authentication over again by requesting
        'none' authentication.
        """
    authClient = ClientAuthWithoutPrivateKey(b'foo', FakeTransport.Service())
    authClient.transport = FakeTransport(None)
    authClient.transport.sessionID = b'test'
    authClient.serviceStarted()
    authClient.tryAuth(b'publickey')
    authClient.transport.packets = []
    self.assertIsNone(authClient.ssh_USERAUTH_PK_OK(b''))
    self.assertEqual(authClient.transport.packets, [(userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'none'))])