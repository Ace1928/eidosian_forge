import struct
from twisted.conch.ssh import channel, common
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import HostnameEndpoint, connectProtocol
def unpackGlobal_tcpip_forward(data):
    host, rest = common.getNS(data)
    if isinstance(host, bytes):
        host = host.decode('utf-8')
    port = int(struct.unpack('>L', rest[:4])[0])
    return (host, port)