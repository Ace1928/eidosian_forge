import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def pwd(self):
    """Return current working directory."""
    resp = self.voidcmd('PWD')
    if not resp.startswith('257'):
        return ''
    return parse257(resp)