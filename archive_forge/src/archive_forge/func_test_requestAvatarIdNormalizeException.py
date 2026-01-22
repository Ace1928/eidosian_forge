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
def test_requestAvatarIdNormalizeException(self):
    """
        Exceptions raised while verifying the key should be normalized into an
        C{UnauthorizedLogin} failure.
        """

    def _checkKey(ignored):
        return True
    self.patch(self.checker, 'checkKey', _checkKey)
    credentials = SSHPrivateKey(b'test', None, b'blob', b'sigData', b'sig')
    d = self.checker.requestAvatarId(credentials)

    def _verifyLoggedException(failure):
        errors = self.flushLoggedErrors(keys.BadKeyError)
        self.assertEqual(len(errors), 1)
        return failure
    d.addErrback(_verifyLoggedException)
    return self.assertFailure(d, UnauthorizedLogin)