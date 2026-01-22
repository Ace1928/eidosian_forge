import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_CHANNEL_WINDOW_ADJUST(self):
    """
        Test that channel window adjust messages add bytes to the channel
        window.
        """
    channel = TestChannel()
    self._openChannel(channel)
    oldWindowSize = channel.remoteWindowLeft
    self.conn.ssh_CHANNEL_WINDOW_ADJUST(b'\x00\x00\x00\x00\x00\x00\x00\x01')
    self.assertEqual(channel.remoteWindowLeft, oldWindowSize + 1)