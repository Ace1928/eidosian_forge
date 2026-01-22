import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_sendEOF(self):
    """
        Test that channel EOF messages are sent in the right format.
        """
    channel = TestChannel()
    self._openChannel(channel)
    self.conn.sendEOF(channel)
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_EOF, b'\x00\x00\x00\xff')])
    channel.localClosed = True
    self.conn.sendEOF(channel)
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_EOF, b'\x00\x00\x00\xff')])