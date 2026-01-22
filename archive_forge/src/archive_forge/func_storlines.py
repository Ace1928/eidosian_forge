import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def storlines(self, cmd, fp, callback=None):
    """Store a file in line mode.  A new port is created for you.

        Args:
          cmd: A STOR command.
          fp: A file-like object with a readline() method.
          callback: An optional single parameter callable that is called on
                    each line after it is sent.  [default: None]

        Returns:
          The response code.
        """
    self.voidcmd('TYPE A')
    with self.transfercmd(cmd) as conn:
        while 1:
            buf = fp.readline(self.maxline + 1)
            if len(buf) > self.maxline:
                raise Error('got more than %d bytes' % self.maxline)
            if not buf:
                break
            if buf[-2:] != B_CRLF:
                if buf[-1] in B_CRLF:
                    buf = buf[:-1]
                buf = buf + B_CRLF
            conn.sendall(buf)
            if callback:
                callback(buf)
        if _SSLSocket is not None and isinstance(conn, _SSLSocket):
            conn.unwrap()
    return self.voidresp()