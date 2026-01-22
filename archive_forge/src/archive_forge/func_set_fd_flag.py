import fcntl
import os
from functools import partial
from pyudev._ctypeslib.libc import ERROR_CHECKERS, FD_PAIR, SIGNATURES
from pyudev._ctypeslib.utils import load_ctypes_library
def set_fd_flag(fd, flag):
    """Set a flag on a file descriptor.

    ``fd`` is the file descriptor or file object, ``flag`` the flag as integer.

    """
    flags = fcntl.fcntl(fd, fcntl.F_GETFD, 0)
    fcntl.fcntl(fd, fcntl.F_SETFD, flags | flag)