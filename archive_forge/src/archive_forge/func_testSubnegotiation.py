from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testSubnegotiation(self):
    h = self.p.protocol
    cmd = telnet.IAC + telnet.SB + b'\x12hello world' + telnet.IAC + telnet.SE
    L = [b'These are some bytes but soon' + cmd, b'there will be some more']
    for b in L:
        self.p.dataReceived(b)
    self.assertEqual(h.data, b''.join(L).replace(cmd, b''))
    self.assertEqual(h.subcmd, list(iterbytes(b'hello world')))