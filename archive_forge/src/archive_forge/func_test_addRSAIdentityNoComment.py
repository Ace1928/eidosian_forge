import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_addRSAIdentityNoComment(self):
    """
        L{SSHAgentClient.addIdentity} adds the private key it is called
        with to the SSH agent server to which it is connected, associating
        it with the comment it is called with.

        This test asserts that omitting the comment produces an
        empty string for the comment on the server.
        """
    d = self.client.addIdentity(self.rsaPrivate.privateBlob())
    self.pump.flush()

    def _check(ignored):
        serverKey = self.server.factory.keys[self.rsaPrivate.blob()]
        self.assertEqual(self.rsaPrivate, serverKey[0])
        self.assertEqual(b'', serverKey[1])
    return d.addCallback(_check)