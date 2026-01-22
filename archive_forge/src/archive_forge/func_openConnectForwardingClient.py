import struct
from twisted.conch.ssh import channel, common
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import HostnameEndpoint, connectProtocol
def openConnectForwardingClient(remoteWindow, remoteMaxPacket, data, avatar):
    remoteHP, origHP = unpackOpen_direct_tcpip(data)
    return SSHConnectForwardingChannel(remoteHP, remoteWindow=remoteWindow, remoteMaxPacket=remoteMaxPacket, avatar=avatar)