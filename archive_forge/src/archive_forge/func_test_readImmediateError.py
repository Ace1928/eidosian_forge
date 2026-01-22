from __future__ import annotations
import socket
from twisted.internet import udp
from twisted.internet.protocol import DatagramProtocol
from twisted.python.runtime import platformType
from twisted.trial import unittest
def test_readImmediateError(self) -> None:
    """
        If the socket is unconnected, socket reads with an immediate
        connection refusal are ignored, and reading stops. The protocol's
        C{connectionRefused} method is not called.
        """
    udp._sockErrReadRefuse.append(-6000)
    self.addCleanup(udp._sockErrReadRefuse.remove, -6000)
    protocol = KeepReads()
    protocol.connectionRefused = lambda: 1 / 0
    port = udp.Port(None, protocol)
    port.socket = StringUDPSocket([b'a', socket.error(-6000), b'b', socket.error(EWOULDBLOCK)])
    port.doRead()
    self.assertEqual(protocol.reads, [b'a'])
    port.doRead()
    self.assertEqual(protocol.reads, [b'a', b'b'])