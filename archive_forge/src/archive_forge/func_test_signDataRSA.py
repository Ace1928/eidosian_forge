import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_signDataRSA(self):
    """
        Sign data with an RSA private key and then verify it with the public
        key.
        """
    d = self.client.signData(self.rsaPublic.blob(), b'John Hancock')
    self.pump.flush()
    signature = self.successResultOf(d)
    expected = self.rsaPrivate.sign(b'John Hancock')
    self.assertEqual(expected, signature)
    self.assertTrue(self.rsaPublic.verify(signature, b'John Hancock'))