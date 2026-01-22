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
def test_toStrCallsToMessage(self):
    """
        L{dns._EDNSMessage.toStr} calls L{dns._EDNSMessage._toMessage}
        """
    m = dns._EDNSMessage()

    def fakeToMessage(*args, **kwargs):
        raise RaisedArgs(args, kwargs)
    m._toMessage = fakeToMessage
    e = self.assertRaises(RaisedArgs, m.toStr)
    self.assertEqual(((), {}), (e.args, e.kwargs))