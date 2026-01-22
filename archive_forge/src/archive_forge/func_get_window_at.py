from __future__ import with_statement
from winappdbg import win32
from winappdbg.registry import Registry
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses, DebugRegister, \
from winappdbg.process import _ProcessContainer
from winappdbg.window import Window
import sys
import os
import ctypes
import warnings
from os import path, getenv
@staticmethod
def get_window_at(x, y):
    """
        Get the window located at the given coordinates in the desktop.
        If no such window exists an exception is raised.

        @see: L{find_window}

        @type  x: int
        @param x: Horizontal coordinate.
        @type  y: int
        @param y: Vertical coordinate.

        @rtype:  L{Window}
        @return: Window at the requested position. If no such window
            exists a C{WindowsError} exception is raised.

        @raise WindowsError: An error occured while processing this request.
        """
    return Window(win32.WindowFromPoint((x, y)))