from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_disableLocal(self):
    """
        It is an error for L{telnet.Telnet.disableLocal} to be called, since
        L{telnet.Telnet.enableLocal} will never allow any options to be enabled
        locally.  If a subclass overrides enableLocal, it must also override
        disableLocal.
        """
    self.assertRaises(NotImplementedError, self.protocol.disableLocal, b'\x00')