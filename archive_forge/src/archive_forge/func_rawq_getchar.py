import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def rawq_getchar(self):
    """Get next char from raw queue.

        Block if no data is immediately available.  Raise EOFError
        when connection is closed.

        """
    if not self.rawq:
        self.fill_rawq()
        if self.eof:
            raise EOFError
    c = self.rawq[self.irawq:self.irawq + 1]
    self.irawq = self.irawq + 1
    if self.irawq >= len(self.rawq):
        self.rawq = b''
        self.irawq = 0
    return c