import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def sock_avail(self):
    """Test whether data is available on the socket."""
    with _TelnetSelector() as selector:
        selector.register(self, selectors.EVENT_READ)
        return bool(selector.select(0))