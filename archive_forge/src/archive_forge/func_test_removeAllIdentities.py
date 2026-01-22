import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_removeAllIdentities(self):
    """
        Assert that we can remove all identities.
        """
    d = self.client.removeAllIdentities()
    self.pump.flush()

    def _check(ignored):
        self.assertEqual(0, len(self.server.factory.keys))
    return d.addCallback(_check)