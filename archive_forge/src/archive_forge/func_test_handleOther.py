from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_handleOther(self):
    """
        L{server.DNSServerFactory.handleOther} triggers the sending of a
        response message with C{rCode} set to L{dns.ENOTIMP}.
        """
    f = server.DNSServerFactory()
    e = self.assertRaises(RaisingProtocol.WriteMessageArguments, f.handleOther, message=dns.Message(), protocol=RaisingProtocol(), address=None)
    (message,), kwargs = e.args
    self.assertEqual(message.rCode, dns.ENOTIMP)