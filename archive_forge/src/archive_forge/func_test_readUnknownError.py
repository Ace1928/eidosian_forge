from __future__ import annotations
import socket
from twisted.internet import udp
from twisted.internet.protocol import DatagramProtocol
from twisted.python.runtime import platformType
from twisted.trial import unittest
def test_readUnknownError(self) -> None:
    """
        Socket reads with an unknown socket error are raised.
        """
    protocol = KeepReads()
    port = udp.Port(None, protocol)
    port.socket = StringUDPSocket([b'good', socket.error(-1337)])
    self.assertRaises(socket.error, port.doRead)
    self.assertEqual(protocol.reads, [b'good'])