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
def test_defaultCheckers(self):
    """
        L{UNIXPasswordDatabase} with no arguments has checks the C{pwd} database
        and then the C{spwd} database.
        """
    checker = checkers.UNIXPasswordDatabase()

    def crypted(username, password):
        salt = crypt.crypt(password, username)
        crypted = crypt.crypt(password, '$1$' + salt)
        return crypted
    pwd = UserDatabase()
    pwd.addUser('alice', crypted('alice', 'password'), 1, 2, 'foo', '/foo', '/bin/sh')
    pwd.addUser('bob', 'x', 1, 2, 'bar', '/bar', '/bin/sh')
    spwd = ShadowDatabase()
    spwd.addUser('alice', 'wrong', 1, 2, 3, 4, 5, 6, 7)
    spwd.addUser('bob', crypted('bob', 'password'), 8, 9, 10, 11, 12, 13, 14)
    self.patch(checkers, 'pwd', pwd)
    self.patch(checkers, 'spwd', spwd)
    mockos = MockOS()
    self.patch(util, 'os', mockos)
    mockos.euid = 2345
    mockos.egid = 1234
    cred = UsernamePassword(b'alice', b'password')
    self.assertLoggedIn(checker.requestAvatarId(cred), b'alice')
    self.assertEqual(mockos.seteuidCalls, [])
    self.assertEqual(mockos.setegidCalls, [])
    cred.username = b'bob'
    self.assertLoggedIn(checker.requestAvatarId(cred), b'bob')
    self.assertEqual(mockos.seteuidCalls, [0, 2345])
    self.assertEqual(mockos.setegidCalls, [0, 1234])