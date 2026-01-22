import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@staticmethod
def do_ranges_intersect(begin, end, old_begin, old_end):
    """
        Determine if the two given memory address ranges intersect.

        @type  begin: int
        @param begin: Start address of the first range.

        @type  end: int
        @param end: End address of the first range.

        @type  old_begin: int
        @param old_begin: Start address of the second range.

        @type  old_end: int
        @param old_end: End address of the second range.

        @rtype:  bool
        @return: C{True} if the two ranges intersect, C{False} otherwise.
        """
    return old_begin <= begin < old_end or old_begin < end <= old_end or begin <= old_begin < end or (begin < old_end <= end)