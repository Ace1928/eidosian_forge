import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def listener(self):
    """Helper for mt_interact() -- this executes in the other thread."""
    while 1:
        try:
            data = self.read_eager()
        except EOFError:
            print('*** Connection closed by remote host ***')
            return
        if data:
            sys.stdout.write(data.decode('ascii'))
        else:
            sys.stdout.flush()