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
def test_days(self):
    """
        Like L{test_seconds}, but for the C{"D"} suffix which multiplies the
        time value by C{86400}, the number of seconds in a day.
        """
    self.assertEqual(4 * 86400, dns.str2time('4D'))