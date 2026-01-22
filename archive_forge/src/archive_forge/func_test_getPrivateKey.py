import sys
from unittest import skipIf
from twisted.conch.error import ConchError
from twisted.conch.test import keydata
from twisted.internet.testing import StringTransport
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_getPrivateKey(self):
    """
        L{SSHUserAuthClient.getPrivateKey} will load a private key from the
        last used file populated by L{SSHUserAuthClient.getPublicKey}, and
        return a L{Deferred} which fires with the corresponding private L{Key}.
        """
    rsaPrivate = Key.fromString(keydata.privateRSA_openssh)
    options = ConchOptions()
    options.identitys = [self.rsaFile.path]
    client = SSHUserAuthClient(b'user', options, None)
    client.getPublicKey()

    def _cbGetPrivateKey(key):
        self.assertFalse(key.isPublic())
        self.assertEqual(key, rsaPrivate)
    return client.getPrivateKey().addCallback(_cbGetPrivateKey)