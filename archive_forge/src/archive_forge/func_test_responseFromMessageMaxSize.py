from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_responseFromMessageMaxSize(self):
    """
        L{server.DNSServerFactory._responseFromMessage} generates a response
        message whose C{maxSize} attribute has the same value as that found
        on the request.
        """
    factory = server.DNSServerFactory()
    request = dns.Message()
    request.maxSize = 0
    response = factory._responseFromMessage(message=request)
    self.assertEqual(request.maxSize, response.maxSize)