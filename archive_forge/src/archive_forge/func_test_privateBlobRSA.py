import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_privateBlobRSA(self):
    """
        L{keys.Key.privateBlob} returns the SSH protocol-level format of an
        RSA private key.
        """
    numbers = self.rsaObj.private_numbers()
    self.assertEqual(keys.Key(self.rsaObj).privateBlob(), common.NS(b'ssh-rsa') + common.MP(numbers.public_numbers.n) + common.MP(numbers.public_numbers.e) + common.MP(numbers.d) + common.MP(numbers.iqmp) + common.MP(numbers.p) + common.MP(numbers.q))