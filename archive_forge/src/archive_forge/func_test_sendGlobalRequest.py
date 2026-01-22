import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_sendGlobalRequest(self):
    """
        Test that global request messages are sent in the right format.
        """
    d = self.conn.sendGlobalRequest(b'wantReply', b'data', True)
    d.addErrback(lambda failure: None)
    self.conn.sendGlobalRequest(b'noReply', b'', False)
    self.assertEqual(self.transport.packets, [(connection.MSG_GLOBAL_REQUEST, common.NS(b'wantReply') + b'\xffdata'), (connection.MSG_GLOBAL_REQUEST, common.NS(b'noReply') + b'\x00')])
    self.assertEqual(self.conn.deferreds, {'global': [d]})