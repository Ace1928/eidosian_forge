from __future__ import annotations
import argparse
import os
import signal
import sys
import threading
from .sync.client import ClientConnection, connect
from .version import version as websockets_version
def win_enable_vt100() -> None:
    """
        Enable VT-100 for console output on Windows.

        See also https://bugs.python.org/issue29059.

        """
    import ctypes
    STD_OUTPUT_HANDLE = ctypes.c_uint(-11)
    INVALID_HANDLE_VALUE = ctypes.c_uint(-1)
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 4
    handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    if handle == INVALID_HANDLE_VALUE:
        raise RuntimeError('unable to obtain stdout handle')
    cur_mode = ctypes.c_uint()
    if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(cur_mode)) == 0:
        raise RuntimeError('unable to query current console mode')
    py_int_mode = int.from_bytes(cur_mode, sys.byteorder)
    new_mode = ctypes.c_uint(py_int_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    if ctypes.windll.kernel32.SetConsoleMode(handle, new_mode) == 0:
        raise RuntimeError('unable to set console mode')