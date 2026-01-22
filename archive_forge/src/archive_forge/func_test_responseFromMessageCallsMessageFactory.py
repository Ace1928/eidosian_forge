from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_responseFromMessageCallsMessageFactory(self):
    """
        L{server.DNSServerFactory._responseFromMessage} calls
        C{dns._responseFromMessage} to generate a response
        message from the request message. It supplies the request message and
        other keyword arguments which should be passed to the response message
        initialiser.
        """
    factory = server.DNSServerFactory()
    self.patch(dns, '_responseFromMessage', raiser)
    request = dns.Message()
    e = self.assertRaises(RaisedArguments, factory._responseFromMessage, message=request, rCode=dns.OK)
    self.assertEqual(((), dict(responseConstructor=factory._messageFactory, message=request, rCode=dns.OK, recAv=factory.canRecurse, auth=False)), (e.args, e.kwargs))