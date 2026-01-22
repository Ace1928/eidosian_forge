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
def test_registerCheckerWithInterface(self):
    """
        If a specific interface is passed into
        L{SSHProtocolChecker.registerChecker}, that interface should be
        registered instead of what the checker specifies in
        credentialIntefaces.
        """
    checker = checkers.SSHProtocolChecker()
    self.assertEqual(checker.credentialInterfaces, [])
    checker.registerChecker(checkers.SSHPublicKeyDatabase(), IUsernamePassword)
    self.assertEqual(checker.credentialInterfaces, [IUsernamePassword])
    self.assertIsInstance(checker.checkers[IUsernamePassword], checkers.SSHPublicKeyDatabase)