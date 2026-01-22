from __future__ import annotations
import sys
from ctypes import pointer
from ..utils import SPHINX_AUTODOC_RUNNING
from ctypes.wintypes import BOOL, DWORD, HANDLE
from prompt_toolkit.win32_types import SECURITY_ATTRIBUTES
def create_win32_event() -> HANDLE:
    """
    Creates a Win32 unnamed Event .
    http://msdn.microsoft.com/en-us/library/windows/desktop/ms682396(v=vs.85).aspx
    """
    return HANDLE(windll.kernel32.CreateEventA(pointer(SECURITY_ATTRIBUTES()), BOOL(True), BOOL(False), None))