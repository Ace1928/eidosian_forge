import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@staticmethod
def path_is_relative(path):
    """
        @see: L{path_is_absolute}

        @type  path: str
        @param path: Absolute or relative path.

        @rtype:  bool
        @return: C{True} if the path is relative, C{False} if it's absolute.
        """
    return win32.PathIsRelative(path)