from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
def enter_alternate_screen(self):
    """
        Go to alternate screen buffer.
        """
    if not self._in_alternate_screen:
        GENERIC_READ = 2147483648
        GENERIC_WRITE = 1073741824
        handle = self._winapi(windll.kernel32.CreateConsoleScreenBuffer, GENERIC_READ | GENERIC_WRITE, DWORD(0), None, DWORD(1), None)
        self._winapi(windll.kernel32.SetConsoleActiveScreenBuffer, handle)
        self.hconsole = handle
        self._in_alternate_screen = True