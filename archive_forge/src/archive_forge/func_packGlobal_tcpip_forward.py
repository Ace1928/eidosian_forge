import struct
from twisted.conch.ssh import channel, common
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import HostnameEndpoint, connectProtocol
def packGlobal_tcpip_forward(peer):
    """
    Pack the data for tcpip forwarding.

    @param peer: A tuple of the (host, port) .
    @type peer: L{tuple}
    """
    host, port = peer
    return common.NS(host) + struct.pack('>L', port)