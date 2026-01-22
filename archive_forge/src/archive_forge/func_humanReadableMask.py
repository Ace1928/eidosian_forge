import os
import struct
from twisted.internet import fdesc
from twisted.internet.abstract import FileDescriptor
from twisted.python import _inotify, log
def humanReadableMask(mask):
    """
    Auxiliary function that converts a hexadecimal mask into a series
    of human readable flags.
    """
    s = []
    for k, v in _FLAG_TO_HUMAN:
        if k & mask:
            s.append(v)
    return s