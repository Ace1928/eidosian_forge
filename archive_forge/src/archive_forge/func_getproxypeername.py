import base64
import socket
import struct
import sys
def getproxypeername(self):
    """getproxypeername() -> address info
        Returns the IP and port number of the proxy.
        """
    return _orgsocket.getpeername(self)