import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_sendExtendedData(self):
    """
        Test that channel extended data messages are sent in the right format.
        """
    channel = TestChannel()
    self._openChannel(channel)
    self.conn.sendExtendedData(channel, 1, b'test')
    channel.localClosed = True
    self.conn.sendExtendedData(channel, 2, b'test2')
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_EXTENDED_DATA, b'\x00\x00\x00\xff' + b'\x00\x00\x00\x01' + common.NS(b'test'))])