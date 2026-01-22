import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def read_some(self):
    """Read at least one byte of cooked data unless EOF is hit.

        Return b'' if EOF is hit.  Block if no data is immediately
        available.

        """
    self.process_rawq()
    while not self.cookedq and (not self.eof):
        self.fill_rawq()
        self.process_rawq()
    buf = self.cookedq
    self.cookedq = b''
    return buf