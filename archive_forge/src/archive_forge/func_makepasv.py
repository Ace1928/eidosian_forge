import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def makepasv(self):
    """Internal: Does the PASV or EPSV handshake -> (address, port)"""
    if self.af == socket.AF_INET:
        untrusted_host, port = parse227(self.sendcmd('PASV'))
        if self.trust_server_pasv_ipv4_address:
            host = untrusted_host
        else:
            host = self.sock.getpeername()[0]
    else:
        host, port = parse229(self.sendcmd('EPSV'), self.sock.getpeername())
    return (host, port)