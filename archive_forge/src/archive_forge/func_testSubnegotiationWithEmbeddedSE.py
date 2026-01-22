from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testSubnegotiationWithEmbeddedSE(self):
    h = self.p.protocol
    cmd = telnet.IAC + telnet.SB + b'\x12' + telnet.SE + telnet.IAC + telnet.SE
    L = [b'Some bytes are here' + cmd + b'and here', b'and here']
    for b in L:
        self.p.dataReceived(b)
    self.assertEqual(h.data, b''.join(L).replace(cmd, b''))
    self.assertEqual(h.subcmd, [telnet.SE])