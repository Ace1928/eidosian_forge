import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_privateBlobEC(self):
    """
        L{keys.Key.privateBlob} returns the SSH ptotocol-level format of EC
        private key.
        """
    from cryptography.hazmat.primitives import serialization
    self.assertEqual(keys.Key(self.ecObj).privateBlob(), common.NS(keydata.ECDatanistp256['curve']) + common.NS(keydata.ECDatanistp256['curve'][-8:]) + common.NS(self.ecObj.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)) + common.MP(self.ecObj.private_numbers().private_value))