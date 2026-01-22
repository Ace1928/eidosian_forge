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
def test_loopThroughFunctions(self):
    """
        UNIXPasswordDatabase.requestAvatarId loops through each getpwnam
        function associated with it and returns a L{Deferred} which fires with
        the result of the first one which returns a value other than None.
        ones do not verify the password.
        """

    def verifyCryptedPassword(crypted, pw):
        return crypted == pw

    def getpwnam1(username):
        return [username, 'not the password']

    def getpwnam2(username):
        return [username, 'password']
    self.patch(checkers, 'verifyCryptedPassword', verifyCryptedPassword)
    checker = checkers.UNIXPasswordDatabase([getpwnam1, getpwnam2])
    credential = UsernamePassword(b'username', b'password')
    self.assertLoggedIn(checker.requestAvatarId(credential), b'username')