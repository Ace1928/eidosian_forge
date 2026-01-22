import os
from unittest import skipIf
from twisted.conch.ssh._kex import getDHGeneratorAndPrime
from twisted.conch.test import keydata
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
def test_getPrivateKeys(self) -> None:
    """
        Will return the available private keys in the data directory, ignoring
        key files which failed to be loaded.
        """
    keys = self.factory.getPrivateKeys()
    self.assertEqual(len(keys), 2)
    keyTypes = keys.keys()
    self.assertEqual(set(keyTypes), {b'ssh-rsa', b'ssh-dss'})
    self.assertEqual(self.mockos.seteuidCalls, [])
    self.assertEqual(self.mockos.setegidCalls, [])