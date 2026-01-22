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
def test_decodeOnlyExpectedBytes(self):
    """
        L{dns._OPTHeader.decode} reads only the bytes from the current
        file position to the end of the record that is being
        decoded. Trailing bytes are not consumed.
        """
    b = BytesIO(OPTNonStandardAttributes.bytes() + b'xxxx')
    decodedHeader = dns._OPTHeader()
    decodedHeader.decode(b)
    self.assertEqual(b.tell(), len(b.getvalue()) - len(b'xxxx'))