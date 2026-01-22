from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testRefuseWill(self):
    cmd = telnet.IAC + telnet.WILL + b'\x12'
    data = b'surrounding bytes' + cmd + b'to spice things up'
    self.p.dataReceived(data)
    self.assertEqual(self.p.protocol.data, data.replace(cmd, b''))
    self.assertEqual(self.t.value(), telnet.IAC + telnet.DONT + b'\x12')
    self._enabledHelper(self.p.protocol)