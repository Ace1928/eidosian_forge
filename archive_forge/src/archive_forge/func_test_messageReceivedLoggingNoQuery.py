from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_messageReceivedLoggingNoQuery(self):
    """
        L{server.DNSServerFactory.messageReceived} logs about an empty query if
        the message had no queries and C{verbose} is C{>0}.
        """
    m = dns.Message()
    f = NoResponseDNSServerFactory(verbose=1)
    assertLogMessage(self, ["Empty query from ('192.0.2.100', 53)"], f.messageReceived, message=m, proto=None, address=('192.0.2.100', 53))