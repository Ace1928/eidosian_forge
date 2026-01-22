import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def prot_p(self):
    """Set up secure data connection."""
    self.voidcmd('PBSZ 0')
    resp = self.voidcmd('PROT P')
    self._prot_p = True
    return resp