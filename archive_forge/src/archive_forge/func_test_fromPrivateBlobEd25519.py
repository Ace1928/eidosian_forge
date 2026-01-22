import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
@skipWithoutEd25519
def test_fromPrivateBlobEd25519(self):
    """
        A private Ed25519 key is correctly generated from a private key blob.
        """
    ed25519Blob = common.NS(b'ssh-ed25519') + common.NS(keydata.Ed25519Data['a']) + common.NS(keydata.Ed25519Data['k'] + keydata.Ed25519Data['a'])
    ed25519Key = keys.Key._fromString_PRIVATE_BLOB(ed25519Blob)
    self.assertFalse(ed25519Key.isPublic())
    self.assertEqual(keydata.Ed25519Data, ed25519Key.data())
    self.assertEqual(ed25519Key, keys.Key._fromString_PRIVATE_BLOB(ed25519Key.privateBlob()))