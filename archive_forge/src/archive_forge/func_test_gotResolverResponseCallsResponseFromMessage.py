from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_gotResolverResponseCallsResponseFromMessage(self):
    """
        L{server.DNSServerFactory.gotResolverResponse} calls
        L{server.DNSServerFactory._responseFromMessage} to generate a response.
        """
    factory = NoResponseDNSServerFactory()
    factory._responseFromMessage = raiser
    request = dns.Message()
    request.timeReceived = 1
    e = self.assertRaises(RaisedArguments, factory.gotResolverResponse, ([], [], []), protocol=None, message=request, address=None)
    self.assertEqual(((), dict(message=request, rCode=dns.OK, answers=[], authority=[], additional=[])), (e.args, e.kwargs))