import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
def test_truncatedPacket(self):
    """
        Test that when a short datagram is received, datagramReceived does
        not raise an exception while processing it.
        """
    self.proto.datagramReceived(b'', address.IPv4Address('UDP', '127.0.0.1', 12345))
    self.assertEqual(self.controller.messages, [])