from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_responseFromMessageNewMessage(self):
    """
        L{server.DNSServerFactory._responseFromMessage} generates a response
        message which is a copy of the request message.
        """
    factory = server.DNSServerFactory()
    request = dns.Message(answer=False, recAv=False)
    response = (factory._responseFromMessage(message=request),)
    self.assertIsNot(request, response)