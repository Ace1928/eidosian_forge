import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_getChannelWithAvatar(self):
    """
        Test that getChannel dispatches to the avatar when an avatar is
        present. Correct functioning without the avatar is verified in
        test_CHANNEL_OPEN.
        """
    channel = self.conn.getChannel(b'TestChannel', 50, 30, b'data')
    self.assertEqual(channel.data, b'data')
    self.assertEqual(channel.remoteWindowLeft, 50)
    self.assertEqual(channel.remoteMaxPacket, 30)
    self.assertRaises(error.ConchError, self.conn.getChannel, b'BadChannel', 50, 30, b'data')