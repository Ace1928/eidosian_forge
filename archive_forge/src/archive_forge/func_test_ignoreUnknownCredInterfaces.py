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
def test_ignoreUnknownCredInterfaces(self):
    """
        L{SSHUserAuthServer} sets up
        C{SSHUserAuthServer.supportedAuthentications} by checking the portal's
        credentials interfaces and mapping them to SSH authentication method
        strings.  If the Portal advertises an interface that
        L{SSHUserAuthServer} can't map, it should be ignored.  This is a white
        box test.
        """
    server = userauth.SSHUserAuthServer()
    server.transport = FakeTransport(self.portal)
    self.portal.registerChecker(AnonymousChecker())
    server.serviceStarted()
    server.serviceStopped()
    server.supportedAuthentications.sort()
    self.assertEqual(server.supportedAuthentications, [b'password', b'publickey'])