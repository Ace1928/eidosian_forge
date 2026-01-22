from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testBoundarySubnegotiation(self):
    cmd = telnet.IAC + telnet.SB + b'\x12' + telnet.SE + b'hello' + telnet.IAC + telnet.SE
    for i in range(len(cmd)):
        h = self.p.protocol = TestProtocol()
        h.makeConnection(self.p)
        a, b = (cmd[:i], cmd[i:])
        L = [b'first part' + a, b + b'last part']
        for data in L:
            self.p.dataReceived(data)
        self.assertEqual(h.data, b''.join(L).replace(cmd, b''))
        self.assertEqual(h.subcmd, [telnet.SE] + list(iterbytes(b'hello')))