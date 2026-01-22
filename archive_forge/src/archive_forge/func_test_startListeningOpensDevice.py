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
def test_startListeningOpensDevice(self):
    """
        L{TuntapPort.startListening} opens the tunnel factory character special
        device C{"/dev/net/tun"} and configures it as a I{tun} tunnel.
        """
    system = self.system
    self.port.startListening()
    tunnel = self.system.getTunnel(self.port)
    expected = (system.O_RDWR | system.O_CLOEXEC | system.O_NONBLOCK, b'tun0' + b'\x00' * (_IFNAMSIZ - len(b'tun0')), self.port.interface, False, True)
    actual = (tunnel.openFlags, tunnel.requestedName, tunnel.name, tunnel.blocking, tunnel.closeOnExec)
    self.assertEqual(expected, actual)