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
def test_receiveMessageNotInLiveMessages(self):
    """
        When receiving a message whose id is not in L{DNSProtocol.liveMessages}
        the message will be received by L{DNSProtocol.controller}.
        """
    message = dns.Message()
    message.id = 1
    message.answers = [dns.RRHeader(payload=dns.Record_A(address='1.2.3.4'))]
    string = message.toStr()
    string = struct.pack('!H', len(string)) + string
    self.proto.dataReceived(string)
    self.assertEqual(self.controller.messages[-1][0].toStr(), message.toStr())