from binascii import hexlify
import configparser
from contextlib import contextmanager
from fcntl import ioctl
import os
import struct
import uuid
from os_brick import exception
from os_brick import privileged
@contextmanager
def open_scini_device():
    """Open scini device for low-level I/O using contextmanager.

    File descriptor will be closed after all operations performed if it was
    opened successfully.

    :return: scini device file descriptor
    :rtype: int
    """
    fd = None
    try:
        fd = os.open(SCINI_DEVICE_PATH, os.O_RDWR)
        yield fd
    finally:
        if fd:
            os.close(fd)