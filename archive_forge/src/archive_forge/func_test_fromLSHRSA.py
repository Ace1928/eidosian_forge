import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_fromLSHRSA(self):
    """
        RSA public and private keys can be generated from a LSH strings.
        """
    self._testPublicPrivateFromString(keydata.publicRSA_lsh, keydata.privateRSA_lsh, 'RSA', keydata.RSAData)