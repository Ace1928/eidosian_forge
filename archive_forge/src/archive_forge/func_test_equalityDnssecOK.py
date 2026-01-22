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
def test_equalityDnssecOK(self):
    """
        Two L{OPTHeader} instances compare equal if they have the same
        dnssecOK flags.
        """
    self.assertNormalEqualityImplementation(dns._OPTHeader(dnssecOK=True), dns._OPTHeader(dnssecOK=True), dns._OPTHeader(dnssecOK=False))