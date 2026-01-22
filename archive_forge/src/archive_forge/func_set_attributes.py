from __future__ import unicode_literals
from ctypes import windll, byref, ArgumentError, c_char, c_long, c_ulong, c_uint, pointer
from ctypes.wintypes import DWORD
from prompt_toolkit.renderer import Output
from prompt_toolkit.styles import ANSI_COLOR_NAMES
from prompt_toolkit.win32_types import CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE, STD_INPUT_HANDLE, COORD, SMALL_RECT
import os
import six
def set_attributes(self, attrs):
    fgcolor, bgcolor, bold, underline, italic, blink, reverse = attrs
    attrs = self.default_attrs
    if fgcolor is not None:
        attrs = attrs & ~15
        attrs |= self.color_lookup_table.lookup_fg_color(fgcolor)
    if bgcolor is not None:
        attrs = attrs & ~240
        attrs |= self.color_lookup_table.lookup_bg_color(bgcolor)
    if reverse:
        attrs = attrs & ~255 | (attrs & 15) << 4 | (attrs & 240) >> 4
    self._winapi(windll.kernel32.SetConsoleTextAttribute, self.hconsole, attrs)