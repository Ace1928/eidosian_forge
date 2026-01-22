import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
@staticmethod
def path_is_absolute(path):
    """
        @see: L{path_is_relative}

        @type  path: str
        @param path: Absolute or relative path.

        @rtype:  bool
        @return: C{True} if the path is absolute, C{False} if it's relative.
        """
    return not win32.PathIsRelative(path)