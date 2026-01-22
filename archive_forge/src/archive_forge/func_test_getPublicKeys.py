import os
from unittest import skipIf
from twisted.conch.ssh._kex import getDHGeneratorAndPrime
from twisted.conch.test import keydata
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
def test_getPublicKeys(self) -> None:
    """
        L{OpenSSHFactory.getPublicKeys} should return the available public keys
        in the data directory
        """
    keys = self.factory.getPublicKeys()
    self.assertEqual(len(keys), 1)
    keyTypes = keys.keys()
    self.assertEqual(list(keyTypes), [b'ssh-rsa'])