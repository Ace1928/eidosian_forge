from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
def quit_alternate_screen(self):
    """
        Make stdout again the active buffer.
        """
    if self._in_alternate_screen:
        stdout = self._winapi(windll.kernel32.GetStdHandle, STD_OUTPUT_HANDLE)
        self._winapi(windll.kernel32.SetConsoleActiveScreenBuffer, stdout)
        self._winapi(windll.kernel32.CloseHandle, self.hconsole)
        self.hconsole = stdout
        self._in_alternate_screen = False