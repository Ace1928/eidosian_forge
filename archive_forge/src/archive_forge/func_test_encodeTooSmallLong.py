import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def test_encodeTooSmallLong(self):
    """
        Test that a negative long below the implementation-specific limit is
        rejected as too small to be encoded.
        """
    largest = self._getLargest()
    self.assertRaises(banana.BananaError, self.enc.sendEncoded, largest)