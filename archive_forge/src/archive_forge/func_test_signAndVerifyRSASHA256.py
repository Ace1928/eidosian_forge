import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_signAndVerifyRSASHA256(self):
    """
        Signed data can be verified using RSA with SHA-256.
        """
    data = b'some-data'
    key = keys.Key.fromString(keydata.privateRSA_openssh)
    signature = key.sign(data, signatureType=b'rsa-sha2-256')
    self.assertTrue(key.public().verify(signature, data))
    self.assertTrue(key.verify(signature, data))
    signatureType, signature = common.getNS(signature)
    self.assertEqual(signatureType, b'rsa-sha2-256')
    self.assertIsNone(key._keyObject.public_key().verify(common.getNS(signature)[0], data, padding.PKCS1v15(), hashes.SHA256()))