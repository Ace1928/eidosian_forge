from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testAcceptDo(self):
    cmd = telnet.IAC + telnet.DO + b'\x19'
    data = b'padding' + cmd + b'trailer'
    h = self.p.protocol
    h.localEnableable = (b'\x19',)
    self.p.dataReceived(data)
    self.assertEqual(self.t.value(), telnet.IAC + telnet.WILL + b'\x19')
    self._enabledHelper(h, eL=[b'\x19'])