from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_messageReceivedLogging2(self):
    """
        L{server.DNSServerFactory.messageReceived} logs the repr of all queries
        in the message if C{verbose} is set to C{2}.
        """
    m = dns.Message()
    m.addQuery(name='example.com', type=dns.MX)
    m.addQuery(name='example.com', type=dns.AAAA)
    f = NoResponseDNSServerFactory(verbose=2)
    assertLogMessage(self, ["<Query example.com MX IN> <Query example.com AAAA IN> query from ('192.0.2.100', 53)"], f.messageReceived, message=m, proto=None, address=('192.0.2.100', 53))