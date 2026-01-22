import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def retrlines(self, cmd, callback=None):
    """Retrieve data in line mode.  A new port is created for you.

        Args:
          cmd: A RETR, LIST, or NLST command.
          callback: An optional single parameter callable that is called
                    for each line with the trailing CRLF stripped.
                    [default: print_line()]

        Returns:
          The response code.
        """
    if callback is None:
        callback = print_line
    resp = self.sendcmd('TYPE A')
    with self.transfercmd(cmd) as conn, conn.makefile('r', encoding=self.encoding) as fp:
        while 1:
            line = fp.readline(self.maxline + 1)
            if len(line) > self.maxline:
                raise Error('got more than %d bytes' % self.maxline)
            if self.debugging > 2:
                print('*retr*', repr(line))
            if not line:
                break
            if line[-2:] == CRLF:
                line = line[:-2]
            elif line[-1:] == '\n':
                line = line[:-1]
            callback(line)
        if _SSLSocket is not None and isinstance(conn, _SSLSocket):
            conn.unwrap()
    return self.voidresp()