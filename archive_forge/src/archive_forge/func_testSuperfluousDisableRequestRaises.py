from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testSuperfluousDisableRequestRaises(self):
    d = self.p.dont(b'\xab')
    return self.assertFailure(d, telnet.AlreadyDisabled)