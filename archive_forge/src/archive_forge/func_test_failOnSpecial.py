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
def test_failOnSpecial(self):
    """
        If the password returned by any function is C{""}, C{"x"}, or C{"*"} it
        is not compared against the supplied password.  Instead it is skipped.
        """
    pwd = UserDatabase()
    pwd.addUser('alice', '', 1, 2, '', 'foo', 'bar')
    pwd.addUser('bob', 'x', 1, 2, '', 'foo', 'bar')
    pwd.addUser('carol', '*', 1, 2, '', 'foo', 'bar')
    self.patch(checkers, 'pwd', pwd)
    checker = checkers.UNIXPasswordDatabase([checkers._pwdGetByName])
    cred = UsernamePassword(b'alice', b'')
    self.assertUnauthorizedLogin(checker.requestAvatarId(cred))
    cred = UsernamePassword(b'bob', b'x')
    self.assertUnauthorizedLogin(checker.requestAvatarId(cred))
    cred = UsernamePassword(b'carol', b'*')
    self.assertUnauthorizedLogin(checker.requestAvatarId(cred))