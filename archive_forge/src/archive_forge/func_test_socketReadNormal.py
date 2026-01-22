from __future__ import annotations
import socket
from twisted.internet import udp
from twisted.internet.protocol import DatagramProtocol
from twisted.python.runtime import platformType
from twisted.trial import unittest
def test_socketReadNormal(self) -> None:
    """
        Socket reads with some good data followed by a socket error which can
        be ignored causes reading to stop, and no log messages to be logged.
        """
    udp._sockErrReadIgnore.append(-7000)
    self.addCleanup(udp._sockErrReadIgnore.remove, -7000)
    protocol = KeepReads()
    port = udp.Port(None, protocol)
    port.socket = StringUDPSocket([b'result', b'123', socket.error(-7000), b'456', socket.error(-7000)])
    port.doRead()
    self.assertEqual(protocol.reads, [b'result', b'123'])
    port.doRead()
    self.assertEqual(protocol.reads, [b'result', b'123', b'456'])