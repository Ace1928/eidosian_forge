from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
def disable_mouse_support(self):
    ENABLE_MOUSE_INPUT = 16
    handle = windll.kernel32.GetStdHandle(STD_INPUT_HANDLE)
    original_mode = DWORD()
    self._winapi(windll.kernel32.GetConsoleMode, handle, pointer(original_mode))
    self._winapi(windll.kernel32.SetConsoleMode, handle, original_mode.value & ~ENABLE_MOUSE_INPUT)