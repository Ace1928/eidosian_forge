import socket
import struct
from twisted.internet import address, defer
from twisted.internet.error import DNSLookupError
from twisted.protocols import socks
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_badSource(self):
    self.sock.dataReceived(struct.pack('!BBH', 4, 2, 34) + socket.inet_aton('1.2.3.4') + b'fooBAR' + b'\x00')
    sent = self.sock.transport.value()
    self.sock.transport.clear()
    incoming = self.sock.driver_listen.buildProtocol(('1.6.6.6', 666))
    self.assertIsNone(incoming)
    sent = self.sock.transport.value()
    self.sock.transport.clear()
    self.assertEqual(sent, struct.pack('!BBH', 0, 91, 0) + socket.inet_aton('0.0.0.0'))
    self.assertTrue(self.sock.transport.stringTCPTransport_closing)