import errno
import os
import socket
import pytest
from jeepney import FileDescriptor, NoFDError
def assert_not_fd(fd: int):
    """Check that the given number is not open as a file descriptor"""
    with pytest.raises(OSError) as exc_info:
        os.stat(fd)
    assert exc_info.value.errno == errno.EBADF