import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_CHANNEL_REQUEST_failure(self):
    """
        Test that channel requests that fail send MSG_CHANNEL_FAILURE.
        """
    channel = TestChannel()
    self._openChannel(channel)
    d = self.conn.ssh_CHANNEL_REQUEST(b'\x00\x00\x00\x00' + common.NS(b'test') + b'\xff')

    def check(result):
        self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_FAILURE, b'\x00\x00\x00\xff')])
    d.addCallback(self.fail)
    d.addErrback(check)
    return d