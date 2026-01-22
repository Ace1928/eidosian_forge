import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_stopConsuming(self):
    """
        L{policies.ProtocolWrapper.stopConsuming} calls C{stopConsuming} on
        the underlying transport.
        """
    wrapper = self._getWrapper()
    result = []
    wrapper.transport.stopConsuming = lambda: result.append(True)
    wrapper.stopConsuming()
    self.assertEqual(result, [True])