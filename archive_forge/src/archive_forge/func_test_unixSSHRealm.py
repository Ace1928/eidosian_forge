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
def test_unixSSHRealm(self) -> None:
    """
        L{UnixSSHRealm} is an L{IRealm} whose C{.requestAvatar} method returns
        a L{UnixConchUser}.
        """
    userdb = UserDatabase()
    home = '/testing/home/value'
    userdb.addUser('user', home=home)
    self.patch(unix, 'pwd', userdb)
    pwdb = InMemoryUsernamePasswordDatabaseDontUse(user=b'password')
    p = Portal(UnixSSHRealm(), [pwdb])
    creds: IUsernamePassword = UsernamePassword(b'user', b'password')
    result = p.login(creds, None, IConchUser)
    resultInterface, avatar, logout = self.successResultOf(result)
    self.assertIsInstance(avatar, UnixConchUser)
    assert isinstance(avatar, UnixConchUser)
    self.assertEqual(avatar.getHomeDir(), home)