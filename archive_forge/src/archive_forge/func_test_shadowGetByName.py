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
def test_shadowGetByName(self):
    """
        L{_shadowGetByName} returns a tuple of items from the UNIX /etc/shadow
        database if the L{spwd} is present.
        """
    userdb = ShadowDatabase()
    userdb.addUser('bob', 'passphrase', 1, 2, 3, 4, 5, 6, 7)
    self.patch(checkers, 'spwd', userdb)
    self.mockos.euid = 2345
    self.mockos.egid = 1234
    self.patch(util, 'os', self.mockos)
    self.assertEqual(checkers._shadowGetByName('bob'), userdb.getspnam('bob'))
    self.assertEqual(self.mockos.seteuidCalls, [0, 2345])
    self.assertEqual(self.mockos.setegidCalls, [0, 1234])