import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_toOpenSSHDSA(self):
    """
        L{keys.Key.toString} serializes a DSA key in OpenSSH format.
        """
    key = keys.Key.fromString(keydata.privateDSA_lsh)
    self.assertEqual(key.toString('openssh').strip(), keydata.privateDSA_openssh)
    self.assertEqual(key.public().toString('openssh', comment=b'comment'), keydata.publicDSA_openssh)
    self.assertEqual(key.public().toString('openssh'), keydata.publicDSA_openssh[:-8])