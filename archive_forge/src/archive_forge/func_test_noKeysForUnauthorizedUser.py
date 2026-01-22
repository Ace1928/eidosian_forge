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
def test_noKeysForUnauthorizedUser(self):
    """
        If the user is not in the user database provided to
        L{checkers.UNIXAuthorizedKeysFiles}, an empty iterator is returned
        by L{checkers.UNIXAuthorizedKeysFiles.getAuthorizedKeys}.
        """
    keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb, parseKey=lambda x: x)
    self.assertEqual([], list(keydb.getAuthorizedKeys(b'bob')))