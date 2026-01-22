from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_twoApplicationDataBytes(self):
    """
        Two application-data bytes in the default state get delivered
        together.
        """
    self._deliver(b'bc', ('bytes', b'bc'))