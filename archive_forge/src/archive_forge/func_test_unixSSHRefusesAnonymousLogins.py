from zope.interface import implementer
from twisted.conch.interfaces import IConchUser
from twisted.cred.checkers import (
from twisted.cred.credentials import (
from twisted.cred.error import LoginDenied
from twisted.cred.portal import Portal
from twisted.internet.interfaces import IReactorProcess
from twisted.python.fakepwd import UserDatabase
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from .test_session import StubClient, StubConnection
def test_unixSSHRefusesAnonymousLogins(self) -> None:
    """
        L{UnixSSHRealm} will refuse anonymous logins.
        """
    p = Portal(UnixSSHRealm(), [AllowAnonymousAccess()])
    result = p.login(IAnonymous(Anonymous()), None, IConchUser)
    loginDenied = self.failureResultOf(result)
    self.assertIsInstance(loginDenied.value, LoginDenied)