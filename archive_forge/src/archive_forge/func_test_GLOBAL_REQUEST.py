import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_GLOBAL_REQUEST(self):
    """
        Test that global request packets are dispatched to the global_*
        methods and the return values are translated into success or failure
        messages.
        """
    self.conn.ssh_GLOBAL_REQUEST(common.NS(b'TestGlobal') + b'\xff')
    self.assertEqual(self.transport.packets, [(connection.MSG_REQUEST_SUCCESS, b'')])
    self.transport.packets = []
    self.conn.ssh_GLOBAL_REQUEST(common.NS(b'TestData') + b'\xff' + b'test data')
    self.assertEqual(self.transport.packets, [(connection.MSG_REQUEST_SUCCESS, b'test data')])
    self.transport.packets = []
    self.conn.ssh_GLOBAL_REQUEST(common.NS(b'TestBad') + b'\xff')
    self.assertEqual(self.transport.packets, [(connection.MSG_REQUEST_FAILURE, b'')])
    self.transport.packets = []
    self.conn.ssh_GLOBAL_REQUEST(common.NS(b'TestGlobal') + b'\x00')
    self.assertEqual(self.transport.packets, [])