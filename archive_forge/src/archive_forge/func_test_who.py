from twisted.internet import address, defer, protocol, reactor
from twisted.protocols import portforward, wire
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_who(self):
    """
        Test wire.Who protocol.
        """
    t = proto_helpers.StringTransport()
    a = wire.Who()
    a.makeConnection(t)
    self.assertEqual(t.value(), b'root\r\n')