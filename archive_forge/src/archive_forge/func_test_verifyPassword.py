import os
from base64 import encodebytes
from collections import namedtuple
from io import BytesIO
from typing import Optional
from zope.interface.verify import verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet.defer import Deferred
from twisted.python import util
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
def test_verifyPassword(self):
    """
        If the encrypted password provided by the getpwnam function is valid
        (verified by the L{verifyCryptedPassword} function), we callback the
        C{requestAvatarId} L{Deferred} with the username.
        """

    def verifyCryptedPassword(crypted, pw):
        return crypted == pw

    def getpwnam(username):
        return [username, username]
    self.patch(checkers, 'verifyCryptedPassword', verifyCryptedPassword)
    checker = checkers.UNIXPasswordDatabase([getpwnam])
    credential = UsernamePassword(b'username', b'username')
    self.assertLoggedIn(checker.requestAvatarId(credential), b'username')