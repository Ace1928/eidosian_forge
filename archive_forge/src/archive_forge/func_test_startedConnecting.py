import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_startedConnecting(self):
    """
        L{policies.WrappingFactory.startedConnecting} calls
        C{startedConnecting} on the underlying factory.
        """
    result = []

    class Factory:

        def startedConnecting(self, connector):
            result.append(connector)
    wrapper = policies.WrappingFactory(Factory())
    connector = object()
    wrapper.startedConnecting(connector)
    self.assertEqual(result, [connector])