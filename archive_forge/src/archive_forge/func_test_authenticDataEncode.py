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
def test_authenticDataEncode(self):
    """
        L{dns.Message.toStr} encodes L{dns.Message.authenticData} into
        byte4 of the byte string.
        """
    self.assertEqual(dns.Message(authenticData=1).toStr(), MESSAGE_AUTHENTIC_DATA_BYTES)