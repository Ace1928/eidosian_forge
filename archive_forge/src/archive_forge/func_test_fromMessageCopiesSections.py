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
def test_fromMessageCopiesSections(self):
    """
        L{dns._EDNSMessage._fromMessage} returns an L{_EDNSMessage} instance
        whose queries, answers, authority and additional lists are copies (not
        references to) the original message lists.
        """
    standardMessage = dns.Message()
    standardMessage.fromStr(MessageEDNSQuery.bytes())
    ednsMessage = dns._EDNSMessage._fromMessage(standardMessage)
    duplicates = []
    for attrName in ('queries', 'answers', 'authority', 'additional'):
        if getattr(standardMessage, attrName) is getattr(ednsMessage, attrName):
            duplicates.append(attrName)
    if duplicates:
        self.fail('Message and _EDNSMessage shared references to the following section lists after decoding: %s' % (duplicates,))