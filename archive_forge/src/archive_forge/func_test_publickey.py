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
def test_publickey(self):
    """
        Test that the client can authenticate with a public key.
        """
    self.authClient.ssh_USERAUTH_FAILURE(NS(b'publickey') + b'\x00')
    self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'publickey') + b'\x00' + NS(b'ssh-dss') + NS(keys.Key.fromString(keydata.publicDSA_openssh).blob())))
    self.authClient.ssh_USERAUTH_FAILURE(NS(b'publickey') + b'\x00')
    blob = NS(keys.Key.fromString(keydata.publicRSA_openssh).blob())
    self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'publickey') + b'\x00' + NS(b'ssh-rsa') + blob))
    self.authClient.ssh_USERAUTH_PK_OK(NS(b'ssh-rsa') + NS(keys.Key.fromString(keydata.publicRSA_openssh).blob()))
    sigData = NS(self.authClient.transport.sessionID) + bytes((userauth.MSG_USERAUTH_REQUEST,)) + NS(b'foo') + NS(b'nancy') + NS(b'publickey') + b'\x01' + NS(b'ssh-rsa') + blob
    obj = keys.Key.fromString(keydata.privateRSA_openssh)
    self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'publickey') + b'\x01' + NS(b'ssh-rsa') + blob + NS(obj.sign(sigData))))