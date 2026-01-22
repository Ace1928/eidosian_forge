import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_removeRSAIdentity(self):
    """
        Assert that we can remove an RSA identity.
        """
    d = self.client.removeIdentity(self.rsaPrivate.blob())
    self.pump.flush()

    def _check(ignored):
        self.assertEqual(1, len(self.server.factory.keys))
        self.assertIn(self.dsaPrivate.blob(), self.server.factory.keys)
        self.assertNotIn(self.rsaPrivate.blob(), self.server.factory.keys)
    return d.addCallback(_check)