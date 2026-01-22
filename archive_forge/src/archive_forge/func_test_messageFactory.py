from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_messageFactory(self):
    """
        L{server.DNSServerFactory} has a C{_messageFactory} attribute which is
        L{dns.Message} by default.
        """
    self.assertIs(dns.Message, server.DNSServerFactory._messageFactory)