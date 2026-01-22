from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_buildProtocolProtocolOverride(self):
    """
        L{server.DNSServerFactory.buildProtocol} builds a protocol by calling
        L{server.DNSServerFactory.protocol} with its self as a positional
        argument.
        """

    class FakeProtocol:
        factory = None
        args = None
        kwargs = None
    stubProtocol = FakeProtocol()

    def fakeProtocolFactory(*args, **kwargs):
        stubProtocol.args = args
        stubProtocol.kwargs = kwargs
        return stubProtocol
    f = server.DNSServerFactory()
    f.protocol = fakeProtocolFactory
    p = f.buildProtocol(addr=None)
    self.assertEqual((stubProtocol, (f,), {}), (p, p.args, p.kwargs))