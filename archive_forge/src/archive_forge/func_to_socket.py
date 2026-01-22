import array
import os
import socket
from warnings import warn
def to_socket(self):
    """Convert to a socket object

        This returns a standard library :func:`socket.socket` object::

            with fd.to_socket() as sock:
                b = sock.sendall(b'xyz')

        The wrapper object can't be used after calling this. Closing the socket
        object will also close the file descriptor.
        """
    from socket import socket
    self._check()
    s = socket(fileno=self._fd)
    self._fd = self._CONVERTED
    return s