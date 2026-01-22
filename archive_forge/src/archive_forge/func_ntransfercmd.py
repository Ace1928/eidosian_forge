import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def ntransfercmd(self, cmd, rest=None):
    conn, size = super().ntransfercmd(cmd, rest)
    if self._prot_p:
        conn = self.context.wrap_socket(conn, server_hostname=self.host)
    return (conn, size)