import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
@skipWithoutEd25519
def test_toOpenSSHEd25519(self):
    """
        L{keys.Key.toString} serializes an Ed25519 key in OpenSSH's v1 format.
        """
    key = keys.Key.fromString(keydata.privateEd25519_openssh_new)
    new_key_data = key.toString('openssh')
    new_enc_key_data = key.toString('openssh', passphrase='encrypted')
    self.assertEqual(b'-----BEGIN OPENSSH PRIVATE KEY-----', new_key_data.splitlines()[0])
    self.assertEqual(b'-----BEGIN OPENSSH PRIVATE KEY-----', new_enc_key_data.splitlines()[0])
    self.assertEqual(key, keys.Key.fromString(new_key_data))
    self.assertEqual(key, keys.Key.fromString(new_enc_key_data, passphrase='encrypted'))
    self.assertEqual(new_key_data, key.toString('openssh', subtype='v1'))