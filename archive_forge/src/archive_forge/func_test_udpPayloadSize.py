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
def test_udpPayloadSize(self):
    """
        L{dns._OPTHeader.udpPayloadSize} defaults to 4096 as
        recommended in rfc6891 section-6.2.5.
        """
    self.assertEqual(dns._OPTHeader().udpPayloadSize, 4096)