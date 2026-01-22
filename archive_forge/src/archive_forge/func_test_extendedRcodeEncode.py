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
def test_extendedRcodeEncode(self):
    """
        The L(_EDNSMessage.toStr} encodes the extended I{RCODE} (>=16) by
        assigning the lower 4bits to the message RCODE field and the upper 4bits
        to the OPT pseudo record.
        """
    self.assertEqual(self.messageFactory(**MessageEDNSExtendedRCODE.kwargs()).toStr(), MessageEDNSExtendedRCODE.bytes())