import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@staticmethod
def make_absolute(path):
    """
        @type  path: str
        @param path: Relative path.

        @rtype:  str
        @return: Absolute path.
        """
    return win32.GetFullPathName(path)[0]