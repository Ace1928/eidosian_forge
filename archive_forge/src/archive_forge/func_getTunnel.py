import socket
import struct
from collections import deque
from errno import EAGAIN, EBADF, EINTR, EINVAL, ENOBUFS, ENOSYS, EPERM, EWOULDBLOCK
from functools import wraps
from zope.interface import implementer
from twisted.internet.protocol import DatagramProtocol
from twisted.pair.ethernet import EthernetProtocol
from twisted.pair.ip import IPProtocol
from twisted.pair.rawudp import RawUDPProtocol
from twisted.pair.tuntap import _IFNAMSIZ, _TUNSETIFF, TunnelFlags, _IInputOutputSystem
from twisted.python.compat import nativeString
def getTunnel(self, port):
    """
        Get the L{Tunnel} object associated with the given L{TuntapPort}.

        @param port: A L{TuntapPort} previously initialized using this
            L{MemoryIOSystem}.

        @return: The tunnel object created by a prior use of C{open} on this
            object on the tunnel special device file.
        @rtype: L{Tunnel}
        """
    return self._openFiles[port.fileno()]