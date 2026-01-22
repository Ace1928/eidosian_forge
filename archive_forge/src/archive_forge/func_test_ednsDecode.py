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
def test_ednsDecode(self):
    """
        The L(_EDNSMessage} instance created by L{dns._EDNSMessage.fromStr}
        derives its edns specific values (C{ednsVersion}, etc) from the supplied
        OPT record.
        """
    m = self.messageFactory()
    m.fromStr(MessageEDNSComplete.bytes())
    self.assertEqual(m, self.messageFactory(**MessageEDNSComplete.kwargs()))