from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_sendReplyWithAddress(self):
    """
        If L{server.DNSServerFactory.sendReply} is supplied with a protocol
        *and* an address tuple it will supply that address to
        C{protocol.writeMessage}.
        """
    m = dns.Message()
    dummyAddress = object()
    f = server.DNSServerFactory()
    e = self.assertRaises(RaisingProtocol.WriteMessageArguments, f.sendReply, protocol=RaisingProtocol(), message=m, address=dummyAddress)
    args, kwargs = e.args
    self.assertEqual(args, (m, dummyAddress))
    self.assertEqual(kwargs, {})