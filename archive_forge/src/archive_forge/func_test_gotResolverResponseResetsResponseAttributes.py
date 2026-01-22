from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_gotResolverResponseResetsResponseAttributes(self):
    """
        L{server.DNSServerFactory.gotResolverResponse} does not allow request
        attributes to leak into the response ie it sends a response with AD, CD
        set to 0 and none of the records in the request answer sections are
        copied to the response.
        """
    factory = server.DNSServerFactory()
    responses = []
    factory.sendReply = lambda protocol, response, address: responses.append(response)
    request = dns.Message(authenticData=True, checkingDisabled=True)
    request.answers = [object(), object()]
    request.authority = [object(), object()]
    request.additional = [object(), object()]
    factory.gotResolverResponse(([], [], []), protocol=None, message=request, address=None)
    self.assertEqual([dns.Message(rCode=0, answer=True)], responses)