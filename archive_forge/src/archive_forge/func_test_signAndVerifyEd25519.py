import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
@skipWithoutEd25519
def test_signAndVerifyEd25519(self):
    """
        Signed data can be verified using Ed25519.
        """
    data = b'some-data'
    key = keys.Key.fromString(keydata.privateEd25519_openssh_new)
    signature = key.sign(data)
    self.assertTrue(key.public().verify(signature, data))
    self.assertTrue(key.verify(signature, data))