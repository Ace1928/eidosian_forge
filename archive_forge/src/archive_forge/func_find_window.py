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
def find_window(className=None, windowName=None):
    """
        Find the first top-level window in the current desktop to match the
        given class name and/or window name. If neither are provided any
        top-level window will match.

        @see: L{get_window_at}

        @type  className: str
        @param className: (Optional) Class name of the window to find.
            If C{None} or not used any class name will match the search.

        @type  windowName: str
        @param windowName: (Optional) Caption text of the window to find.
            If C{None} or not used any caption text will match the search.

        @rtype:  L{Window} or None
        @return: A window that matches the request. There may be more matching
            windows, but this method only returns one. If no matching window
            is found, the return value is C{None}.

        @raise WindowsError: An error occured while processing this request.
        """
    hWnd = win32.FindWindow(className, windowName)
    if hWnd:
        return Window(hWnd)