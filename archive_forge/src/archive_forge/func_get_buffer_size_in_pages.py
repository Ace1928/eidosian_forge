import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@classmethod
def get_buffer_size_in_pages(cls, address, size):
    """
        Get the number of pages in use by the given buffer.

        @type  address: int
        @param address: Aligned memory address.

        @type  size: int
        @param size: Buffer size.

        @rtype:  int
        @return: Buffer size in number of pages.
        """
    if size < 0:
        size = -size
        address = address - size
    begin, end = cls.align_address_range(address, address + size)
    return int(float(end - begin) / float(cls.pageSize))