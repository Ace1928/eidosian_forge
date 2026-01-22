from ctypes import *
from functools import lru_cache
import unicodedata
from pyglet import compat_platform
import pyglet
from pyglet.window import BaseWindow, WindowException, MouseCursor
from pyglet.window import DefaultMouseCursor, _PlatformEventHandler, _ViewEventHandler
from pyglet.event import EventDispatcher
from pyglet.window import key, mouse
from pyglet.canvas.win32 import Win32Canvas
from pyglet.libs.win32 import _user32, _kernel32, _gdi32, _dwmapi, _shell32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.winkey import *
from pyglet.libs.win32.types import *
def set_exclusive_keyboard(self, exclusive=True):
    if self._exclusive_keyboard == exclusive and self._exclusive_keyboard_focus == self._has_focus:
        return
    if exclusive and self._has_focus:
        _user32.RegisterHotKey(self._hwnd, 0, WIN32_MOD_ALT, VK_TAB)
    elif self._exclusive_keyboard and (not exclusive):
        _user32.UnregisterHotKey(self._hwnd, 0)
    self._exclusive_keyboard = exclusive
    self._exclusive_keyboard_focus = self._has_focus