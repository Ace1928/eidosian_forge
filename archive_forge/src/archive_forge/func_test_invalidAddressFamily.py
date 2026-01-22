import errno
import socket
from zope.interface import verify
from twisted.internet.error import UnsupportedAddressFamily
from twisted.internet.interfaces import IReactorSocket
from twisted.internet.protocol import DatagramProtocol, ServerFactory
from twisted.internet.test.reactormixins import ReactorBuilder, needsRunningReactor
from twisted.python.log import err
from twisted.python.runtime import platform
def test_invalidAddressFamily(self):
    """
        An implementation of L{IReactorSocket.adoptDatagramPort} raises
        L{UnsupportedAddressFamily} if passed an address family it does not
        support.
        """
    reactor = self.buildReactor()
    port = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.addCleanup(port.close)
    arbitrary = 2 ** 16 + 7
    self.assertRaises(UnsupportedAddressFamily, reactor.adoptDatagramPort, port.fileno(), arbitrary, DatagramProtocol())