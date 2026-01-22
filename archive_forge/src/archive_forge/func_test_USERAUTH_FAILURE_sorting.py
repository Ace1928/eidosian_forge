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
def test_USERAUTH_FAILURE_sorting(self):
    """
        ssh_USERAUTH_FAILURE should sort the methods by their position
        in SSHUserAuthClient.preferredOrder.  Methods that are not in
        preferredOrder should be sorted at the end of that list.
        """

    def auth_firstmethod():
        self.authClient.transport.sendPacket(255, b'here is data')

    def auth_anothermethod():
        self.authClient.transport.sendPacket(254, b'other data')
        return True
    self.authClient.auth_firstmethod = auth_firstmethod
    self.authClient.auth_anothermethod = auth_anothermethod
    self.authClient.ssh_USERAUTH_FAILURE(NS(b'anothermethod,password') + b'\x00')
    self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'password') + b'\x00' + NS(b'foo')))
    self.authClient.ssh_USERAUTH_FAILURE(NS(b'firstmethod,anothermethod,password') + b'\xff')
    self.assertEqual(self.authClient.transport.packets[-2:], [(255, b'here is data'), (254, b'other data')])