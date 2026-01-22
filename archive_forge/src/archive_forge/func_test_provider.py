import errno
import socket
from zope.interface import verify
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import IReactorSocket
from twisted.internet.protocol import DatagramProtocol, ServerFactory
from twisted.internet.test.reactormixins import ReactorBuilder, needsRunningReactor
from twisted.python.log import err
from twisted.python.runtime import platform
def test_provider(self):
    """
        The reactor instance returned by C{buildReactor} provides
        L{IReactorSocket}.
        """
    reactor = self.buildReactor()
    self.assertTrue(verify.verifyObject(IReactorSocket, reactor))