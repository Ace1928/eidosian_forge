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
def test_reprFlagsIfSet(self):
    """
        L{dns._EDNSMessage.__repr__} displays flags if they are L{True}.
        """
    m = self.messageFactory(answer=True, auth=True, trunc=True, recDes=True, recAv=True, authenticData=True, checkingDisabled=True, dnssecOK=True)
    self.assertEqual('<_EDNSMessage id=0 flags=answer,auth,trunc,recDes,recAv,authenticData,checkingDisabled,dnssecOK>', repr(m))