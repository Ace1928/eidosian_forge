import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromPrivateBlobECDSA(self):
    """
        A private EC key is correctly generated from a private key blob.
        """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    publicNumbers = ec.EllipticCurvePublicNumbers(x=keydata.ECDatanistp256['x'], y=keydata.ECDatanistp256['y'], curve=ec.SECP256R1())
    ecblob = common.NS(keydata.ECDatanistp256['curve']) + common.NS(keydata.ECDatanistp256['curve'][-8:]) + common.NS(publicNumbers.public_key(default_backend()).public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)) + common.MP(keydata.ECDatanistp256['privateValue'])
    eckey = keys.Key._fromString_PRIVATE_BLOB(ecblob)
    self.assertFalse(eckey.isPublic())
    self.assertEqual(keydata.ECDatanistp256, eckey.data())
    self.assertEqual(eckey, keys.Key._fromString_PRIVATE_BLOB(eckey.privateBlob()))