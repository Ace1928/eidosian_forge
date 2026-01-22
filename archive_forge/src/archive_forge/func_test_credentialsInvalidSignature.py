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
def test_credentialsInvalidSignature(self):
    """
        Calling L{checkers.SSHPublicKeyChecker.requestAvatarId} with
        credentials that are incorrectly signed fails with
        L{UnauthorizedLogin}.
        """
    self.credentials.signature = keys.Key.fromString(keydata.privateDSA_openssh).sign(b'foo')
    self.failureResultOf(self.checker.requestAvatarId(self.credentials), UnauthorizedLogin)