import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromPrivateBlobRSA(self):
    """
        A private RSA key is correctly generated from a private key blob.
        """
    rsaBlob = common.NS(b'ssh-rsa') + common.MP(keydata.RSAData['n']) + common.MP(keydata.RSAData['e']) + common.MP(keydata.RSAData['d']) + common.MP(keydata.RSAData['u']) + common.MP(keydata.RSAData['p']) + common.MP(keydata.RSAData['q'])
    rsaKey = keys.Key._fromString_PRIVATE_BLOB(rsaBlob)
    self.assertFalse(rsaKey.isPublic())
    self.assertEqual(keydata.RSAData, rsaKey.data())
    self.assertEqual(rsaKey, keys.Key._fromString_PRIVATE_BLOB(rsaKey.privateBlob()))