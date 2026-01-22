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
@skipIf(ipv6Skip, ipv6SkipReason)
def test_ipv6AcceptAddress(self):
    """
        Like L{test_ipv4AcceptAddress}, but for IPv6 connections.
        In this case:

          - the first element is C{AF_INET6}
          - the second element is a two-tuple of a hexadecimal IPv6 address
            literal and a port number giving the peer address of the connection
          - the third element is the same type giving the host address of the
            connection
        """
    self._acceptAddressTest(AF_INET6, '::1')