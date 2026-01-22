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
def test_no_publickey(self):
    """
        If there's no public key, auth_publickey should return a Deferred
        called back with a False value.
        """
    self.authClient.getPublicKey = lambda x: None
    d = self.authClient.tryAuth(b'publickey')

    def check(result):
        self.assertFalse(result)
    return d.addCallback(check)