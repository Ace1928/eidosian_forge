import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromOpenSSH(self):
    """
        Test that keys are correctly generated from OpenSSH strings.
        """
    self._testPublicPrivateFromString(keydata.publicECDSA_openssh, keydata.privateECDSA_openssh, 'EC', keydata.ECDatanistp256)
    self._testPublicPrivateFromString(keydata.publicRSA_openssh, keydata.privateRSA_openssh, 'RSA', keydata.RSAData)
    self.assertEqual(keys.Key.fromString(keydata.privateRSA_openssh_encrypted, passphrase=b'encrypted'), keys.Key.fromString(keydata.privateRSA_openssh))
    self._testPublicPrivateFromString(keydata.publicDSA_openssh, keydata.privateDSA_openssh, 'DSA', keydata.DSAData)
    if ED25519_SUPPORTED:
        self._testPublicPrivateFromString(keydata.publicEd25519_openssh, keydata.privateEd25519_openssh_new, 'Ed25519', keydata.Ed25519Data)