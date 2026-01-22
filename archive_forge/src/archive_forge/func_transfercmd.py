import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def transfercmd(self, cmd, rest=None):
    """Like ntransfercmd() but returns only the socket."""
    return self.ntransfercmd(cmd, rest)[0]