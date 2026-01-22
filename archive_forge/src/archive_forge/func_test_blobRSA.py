import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_blobRSA(self):
    """
        Return the over-the-wire SSH format of the RSA public key.
        """
    self.assertEqual(keys.Key(self.rsaObj).blob(), common.NS(b'ssh-rsa') + common.MP(self.rsaObj.private_numbers().public_numbers.e) + common.MP(self.rsaObj.private_numbers().public_numbers.n))