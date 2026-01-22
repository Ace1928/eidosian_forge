import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test__guessStringType(self):
    """
        Test that the _guessStringType method guesses string types
        correctly.
        """
    self.assertEqual(keys.Key._guessStringType(keydata.publicRSA_openssh), 'public_openssh')
    self.assertEqual(keys.Key._guessStringType(keydata.publicDSA_openssh), 'public_openssh')
    self.assertEqual(keys.Key._guessStringType(keydata.publicECDSA_openssh), 'public_openssh')
    if ED25519_SUPPORTED:
        self.assertEqual(keys.Key._guessStringType(keydata.publicEd25519_openssh), 'public_openssh')
    self.assertEqual(keys.Key._guessStringType(keydata.privateRSA_openssh), 'private_openssh')
    self.assertEqual(keys.Key._guessStringType(keydata.privateRSA_openssh_new), 'private_openssh')
    self.assertEqual(keys.Key._guessStringType(keydata.privateDSA_openssh), 'private_openssh')
    self.assertEqual(keys.Key._guessStringType(keydata.privateDSA_openssh_new), 'private_openssh')
    self.assertEqual(keys.Key._guessStringType(keydata.privateECDSA_openssh), 'private_openssh')
    self.assertEqual(keys.Key._guessStringType(keydata.privateECDSA_openssh_new), 'private_openssh')
    if ED25519_SUPPORTED:
        self.assertEqual(keys.Key._guessStringType(keydata.privateEd25519_openssh_new), 'private_openssh')
    self.assertEqual(keys.Key._guessStringType(keydata.publicRSA_lsh), 'public_lsh')
    self.assertEqual(keys.Key._guessStringType(keydata.publicDSA_lsh), 'public_lsh')
    self.assertEqual(keys.Key._guessStringType(keydata.privateRSA_lsh), 'private_lsh')
    self.assertEqual(keys.Key._guessStringType(keydata.privateDSA_lsh), 'private_lsh')
    self.assertEqual(keys.Key._guessStringType(keydata.privateRSA_agentv3), 'agentv3')
    self.assertEqual(keys.Key._guessStringType(keydata.privateDSA_agentv3), 'agentv3')
    self.assertEqual(keys.Key._guessStringType(b'\x00\x00\x00\x07ssh-rsa\x00\x00\x00\x01\x01'), 'blob')
    self.assertEqual(keys.Key._guessStringType(b'\x00\x00\x00\x07ssh-dss\x00\x00\x00\x01\x01'), 'blob')
    self.assertEqual(keys.Key._guessStringType(b'not a key'), None)