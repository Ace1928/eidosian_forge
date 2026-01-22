import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def test_encodeTooLargeLong(self):
    """
        Test that a long above the implementation-specific limit is rejected
        as too large to be encoded.
        """
    smallest = self._getSmallest()
    self.assertRaises(banana.BananaError, self.enc.sendEncoded, smallest)