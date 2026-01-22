import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_isPublic(self):
    """
        The L{keys.Key.isPublic} method returns True for public keys
        otherwise False.
        """
    rsaKey = keys.Key.fromString(keydata.privateRSA_openssh)
    dsaKey = keys.Key.fromString(keydata.privateDSA_openssh)
    ecdsaKey = keys.Key.fromString(keydata.privateECDSA_openssh)
    self.assertTrue(rsaKey.public().isPublic())
    self.assertFalse(rsaKey.isPublic())
    self.assertTrue(dsaKey.public().isPublic())
    self.assertFalse(dsaKey.isPublic())
    self.assertTrue(ecdsaKey.public().isPublic())
    self.assertFalse(ecdsaKey.isPublic())
    if ED25519_SUPPORTED:
        ed25519Key = keys.Key.fromString(keydata.privateEd25519_openssh_new)
        self.assertTrue(ed25519Key.public().isPublic())
        self.assertFalse(ed25519Key.isPublic())