import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_blobDSA(self):
    """
        Return the over-the-wire SSH format of the DSA public key.
        """
    publicNumbers = self.dsaObj.private_numbers().public_numbers
    self.assertEqual(keys.Key(self.dsaObj).blob(), common.NS(b'ssh-dss') + common.MP(publicNumbers.parameter_numbers.p) + common.MP(publicNumbers.parameter_numbers.q) + common.MP(publicNumbers.parameter_numbers.g) + common.MP(publicNumbers.y))