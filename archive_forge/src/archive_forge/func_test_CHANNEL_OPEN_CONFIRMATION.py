import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_CHANNEL_OPEN_CONFIRMATION(self):
    """
        Test that channel open confirmation packets cause the channel to be
        notified that it's open.
        """
    channel = TestChannel()
    self.conn.openChannel(channel)
    self.conn.ssh_CHANNEL_OPEN_CONFIRMATION(b'\x00\x00\x00\x00' * 5)
    self.assertEqual(channel.remoteWindowLeft, 0)
    self.assertEqual(channel.remoteMaxPacket, 0)
    self.assertEqual(channel.specificData, b'\x00\x00\x00\x00')
    self.assertEqual(self.conn.channelsToRemoteChannel[channel], 0)
    self.assertEqual(self.conn.localToRemoteChannel[0], 0)