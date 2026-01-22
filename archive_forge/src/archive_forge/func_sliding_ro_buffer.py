import binascii
import os
import mmap
import sys
import time
import errno
from io import BytesIO
from smmap import (
import hashlib
from gitdb.const import (
def sliding_ro_buffer(filepath, flags=0):
    """
    :return: a buffer compatible object which uses our mapped memory manager internally
        ready to read the whole given filepath"""
    return SlidingWindowMapBuffer(mman.make_cursor(filepath), flags=flags)