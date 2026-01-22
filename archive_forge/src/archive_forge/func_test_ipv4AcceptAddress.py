import errno
import sys
import time
from array import array
from socket import AF_INET, AF_INET6, SOCK_STREAM, SOL_SOCKET, socket
from struct import pack
from unittest import skipIf
from zope.interface.verify import verifyClass
from twisted.internet.interfaces import IPushProducer
from twisted.python.log import msg
from twisted.trial.unittest import TestCase
def test_ipv4AcceptAddress(self):
    """
        L{iocpsupport.get_accept_addrs} returns a three-tuple of address
        information about the socket associated with the file descriptor passed
        to it.  For a connection using IPv4:

          - the first element is C{AF_INET}
          - the second element is a two-tuple of a dotted decimal notation IPv4
            address and a port number giving the peer address of the connection
          - the third element is the same type giving the host address of the
            connection
        """
    self._acceptAddressTest(AF_INET, '127.0.0.1')