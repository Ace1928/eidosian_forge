import socket
import struct
from twisted.internet import address, defer
from twisted.internet.error import DNSLookupError
from twisted.protocols import socks
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_accessDenied(self):
    self.sock.authorize = lambda code, server, port, user: 0
    self.sock.dataReceived(struct.pack('!BBH', 4, 2, 4242) + socket.inet_aton('10.2.3.4') + b'fooBAR' + b'\x00')
    self.assertEqual(self.sock.transport.value(), struct.pack('!BBH', 0, 91, 0) + socket.inet_aton('0.0.0.0'))
    self.assertTrue(self.sock.transport.stringTCPTransport_closing)
    self.assertIsNone(self.sock.driver_listen)