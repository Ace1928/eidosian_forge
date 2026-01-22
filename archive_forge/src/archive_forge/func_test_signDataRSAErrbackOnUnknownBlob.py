import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_signDataRSAErrbackOnUnknownBlob(self):
    """
        Assert that we get an errback if we try to sign data using a key that
        wasn't added.
        """
    del self.server.factory.keys[self.rsaPublic.blob()]
    d = self.client.signData(self.rsaPublic.blob(), b'John Hancock')
    self.pump.flush()
    return self.assertFailure(d, ConchError)