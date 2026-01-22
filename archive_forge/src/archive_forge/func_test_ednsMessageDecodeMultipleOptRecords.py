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
def test_ednsMessageDecodeMultipleOptRecords(self):
    """
        An L(_EDNSMessage} instance created from a byte string containing
        multiple I{OPT} records will discard all the C{OPT} records.

        C{ednsVersion} will be set to L{None}.

        @see: U{https://tools.ietf.org/html/rfc6891#section-6.1.1}
        """
    m = dns.Message()
    m.additional = [dns._OPTHeader(version=2), dns._OPTHeader(version=3)]
    ednsMessage = dns._EDNSMessage()
    ednsMessage.fromStr(m.toStr())
    self.assertIsNone(ednsMessage.ednsVersion)