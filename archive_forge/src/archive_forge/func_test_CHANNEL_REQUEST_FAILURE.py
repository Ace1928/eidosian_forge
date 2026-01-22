import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_CHANNEL_REQUEST_FAILURE(self):
    """
        Test that channel request failure messages cause the Deferred to be
        erred back.
        """
    channel = TestChannel()
    self._openChannel(channel)
    d = self.conn.sendRequest(channel, b'test', b'', True)
    self.conn.ssh_CHANNEL_FAILURE(b'\x00\x00\x00\x00')

    def check(result):
        self.assertEqual(result.value.value, 'channel request failed')
    d.addCallback(self.fail)
    d.addErrback(check)
    return d