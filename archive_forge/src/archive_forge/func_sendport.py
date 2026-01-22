import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def sendport(self, host, port):
    """Send a PORT command with the current host and the given
        port number.
        """
    hbytes = host.split('.')
    pbytes = [repr(port // 256), repr(port % 256)]
    bytes = hbytes + pbytes
    cmd = 'PORT ' + ','.join(bytes)
    return self.voidcmd(cmd)