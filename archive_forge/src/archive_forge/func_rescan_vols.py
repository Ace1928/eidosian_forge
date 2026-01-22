from binascii import hexlify
import configparser
from contextlib import contextmanager
from fcntl import ioctl
import os
import struct
import uuid
from os_brick import exception
from os_brick import privileged
@privileged.default.entrypoint
def rescan_vols(op_code):
    """Rescan ScaleIO volumes via ioctl request.

    :param op_code: operational code
    :type op_code: int
    """
    with open_scini_device() as fd:
        ioctl(fd, op_code, struct.pack('Q', 0))