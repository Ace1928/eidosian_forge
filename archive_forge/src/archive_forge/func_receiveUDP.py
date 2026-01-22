import os
import socket
import struct
from collections import deque
from errno import EAGAIN, EBADF, EINVAL, ENODEV, ENOENT, EPERM, EWOULDBLOCK
from itertools import cycle
from random import randrange
from signal import SIGINT
from typing import Optional
from twisted.python.reflect import ObjectNotFound, namedAny
from zope.interface import Interface, implementer
from zope.interface.verify import verifyObject
from twisted.internet.error import CannotListenError
from twisted.internet.interfaces import IAddress, IListeningPort, IReactorFDSet
from twisted.internet.protocol import (
from twisted.internet.task import Clock
from twisted.pair.ethernet import EthernetProtocol
from twisted.pair.ip import IPProtocol
from twisted.pair.raw import IRawPacketProtocol
from twisted.pair.rawudp import RawUDPProtocol
from twisted.python.compat import iterbytes
from twisted.python.log import addObserver, removeObserver, textFromEventDict
from twisted.python.reflect import fullyQualifiedName
from twisted.trial.unittest import SkipTest, SynchronousTestCase
def receiveUDP(self, fileno, host, port):
    """
        Use the platform network stack to receive a datagram sent to the given
        address.

        @param fileno: The file descriptor of the tunnel used to send the
            datagram.  This is ignored because a real socket is used to receive
            the datagram.
        @type fileno: L{int}

        @param host: The IPv4 address at which the datagram will be received.
        @type host: L{bytes}

        @param port: The UDP port number at which the datagram will be
            received.
        @type port: L{int}

        @return: A L{socket.socket} which can be used to receive the specified
            datagram.
        """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))
    return s