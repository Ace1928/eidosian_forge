from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testAcceptedDisableRequest(self):
    s = self.p.getOptionState(b'B')
    s.him.state = 'yes'
    d = self.p.dont(b'B')
    self.assertEqual(self.t.value(), telnet.IAC + telnet.DONT + b'B')
    self.p.dataReceived(telnet.IAC + telnet.WONT + b'B')
    d.addCallback(self.assertEqual, True)
    d.addCallback(lambda _: self._enabledHelper(self.p.protocol, dR=[b'B']))
    return d