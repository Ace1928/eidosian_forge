import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def test_sendPb(self):
    """
        if pb dialect is selected, the sender must be able to send things in
        that dialect.
        """
    selectDialect(self.enc, b'pb')
    self.enc.sendEncoded(self.vocab)
    self.assertEqual(self.legalPbItem, self.io.getvalue())