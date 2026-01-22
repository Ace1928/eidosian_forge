from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testAcceptWont(self):
    cmd = telnet.IAC + telnet.WONT + b')'
    s = self.p.getOptionState(b')')
    s.him.state = 'yes'
    data = b'fiddle dee' + cmd
    self.p.dataReceived(data)
    self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
    self.assertEqual(self.t.value(), telnet.IAC + telnet.DONT + b')')
    self.assertEqual(s.him.state, 'no')
    self._enabledHelper(self.p.protocol, dR=[b')'])