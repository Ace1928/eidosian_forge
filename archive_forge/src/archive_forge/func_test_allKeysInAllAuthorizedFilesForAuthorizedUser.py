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
def test_allKeysInAllAuthorizedFilesForAuthorizedUser(self):
    """
        If the user is in the user database provided to
        L{checkers.UNIXAuthorizedKeysFiles}, an iterator with all the keys in
        C{~/.ssh/authorized_keys} and C{~/.ssh/authorized_keys2} is returned
        by L{checkers.UNIXAuthorizedKeysFiles.getAuthorizedKeys}.
        """
    self.sshDir.child('authorized_keys2').setContent(b'key 3')
    keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb, parseKey=lambda x: x)
    self.assertEqual(self.expectedKeys + [b'key 3'], list(keydb.getAuthorizedKeys(b'alice')))