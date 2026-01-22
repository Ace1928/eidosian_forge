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
def test_authoritativeMessageDecode(self):
    """
        The message and its L{dns.RRHeader} instances created by C{decode} from
        an authoritative message byte string, are marked as authoritative.
        """
    m = self.messageFactory()
    m.fromStr(MessageAuthoritative.bytes())
    self.assertEqual(m, self.messageFactory(**MessageAuthoritative.kwargs()))