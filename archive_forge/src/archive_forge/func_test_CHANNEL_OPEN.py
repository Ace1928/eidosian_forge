import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_CHANNEL_OPEN(self):
    """
        Test that open channel packets cause a channel to be created and
        opened or a failure message to be returned.
        """
    del self.transport.avatar
    self.conn.ssh_CHANNEL_OPEN(common.NS(b'TestChannel') + b'\x00\x00\x00\x01' * 4)
    self.assertTrue(self.conn.channel.gotOpen)
    self.assertEqual(self.conn.channel.conn, self.conn)
    self.assertEqual(self.conn.channel.data, b'\x00\x00\x00\x01')
    self.assertEqual(self.conn.channel.specificData, b'\x00\x00\x00\x01')
    self.assertEqual(self.conn.channel.remoteWindowLeft, 1)
    self.assertEqual(self.conn.channel.remoteMaxPacket, 1)
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_OPEN_CONFIRMATION, b'\x00\x00\x00\x01\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x80\x00')])
    self.transport.packets = []
    self.conn.ssh_CHANNEL_OPEN(common.NS(b'BadChannel') + b'\x00\x00\x00\x02' * 4)
    self.flushLoggedErrors()
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_OPEN_FAILURE, b'\x00\x00\x00\x02\x00\x00\x00\x03' + common.NS(b'unknown channel') + common.NS(b''))])
    self.transport.packets = []
    self.conn.ssh_CHANNEL_OPEN(common.NS(b'ErrorChannel') + b'\x00\x00\x00\x02' * 4)
    self.flushLoggedErrors()
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_OPEN_FAILURE, b'\x00\x00\x00\x02\x00\x00\x00\x02' + common.NS(b'unknown failure') + common.NS(b''))])