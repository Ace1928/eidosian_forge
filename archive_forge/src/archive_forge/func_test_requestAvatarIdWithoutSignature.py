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
def test_requestAvatarIdWithoutSignature(self):
    """
        L{SSHPublicKeyDatabase.requestAvatarId} should raise L{ValidPublicKey}
        if the credentials represent a valid key without a signature.  This
        tells the user that the key is valid for login, but does not actually
        allow that user to do so without a signature.
        """

    def _checkKey(ignored):
        return True
    self.patch(self.checker, 'checkKey', _checkKey)
    credentials = SSHPrivateKey(b'test', b'ssh-rsa', keydata.publicRSA_openssh, None, None)
    d = self.checker.requestAvatarId(credentials)
    return self.assertFailure(d, ValidPublicKey)