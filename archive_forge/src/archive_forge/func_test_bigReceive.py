from struct import calcsize, pack, unpack
from twisted.protocols.stateful import StatefulProtocol
from twisted.protocols.test import test_basic
from twisted.trial.unittest import TestCase
def test_bigReceive(self):
    r = self.getProtocol()
    big = b''
    for s in self.strings * 4:
        big += pack('!i', len(s)) + s
    r.dataReceived(big)
    self.assertEqual(r.received, self.strings * 4)