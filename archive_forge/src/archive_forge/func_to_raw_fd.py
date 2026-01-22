import array
import os
import socket
from warnings import warn
def to_raw_fd(self):
    """Convert to the low-level integer file descriptor::

            raw_fd = fd.to_raw_fd()
            os.write(raw_fd, b'xyz')
            os.close(raw_fd)

        The :class:`FileDescriptor` can't be used after calling this. The caller
        is responsible for closing the file descriptor.
        """
    self._check()
    self._fd, fd = (self._CONVERTED, self._fd)
    return fd