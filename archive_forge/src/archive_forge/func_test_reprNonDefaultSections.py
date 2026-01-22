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
def test_reprNonDefaultSections(self):
    """
        L{dns.Message.__repr__} displays sections which differ from their
        defaults.
        """
    m = self.messageFactory()
    m.queries = [1, 2, 3]
    m.answers = [4, 5, 6]
    m.authority = [7, 8, 9]
    m.additional = [10, 11, 12]
    self.assertEqual('<_EDNSMessage id=0 queries=[1, 2, 3] answers=[4, 5, 6] authority=[7, 8, 9] additional=[10, 11, 12]>', repr(m))