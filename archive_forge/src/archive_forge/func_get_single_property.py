import locale
import unicodedata
import urllib.parse
from ctypes import *
from functools import lru_cache
from typing import Optional
import pyglet
from pyglet.window import WindowException, MouseCursorException
from pyglet.window import MouseCursor, DefaultMouseCursor, ImageMouseCursor
from pyglet.window import BaseWindow, _PlatformEventHandler, _ViewEventHandler
from pyglet.window import key
from pyglet.window import mouse
from pyglet.event import EventDispatcher
from pyglet.canvas.xlib import XlibCanvas
from pyglet.libs.x11 import xlib
from pyglet.libs.x11 import cursorfont
from pyglet.util import asbytes
def get_single_property(self, window, atom_property, atom_type):
    """ Returns the length, data, and actual atom of a window property. """
    actualAtom = xlib.Atom()
    actualFormat = c_int()
    itemCount = c_ulong()
    bytesAfter = c_ulong()
    data = POINTER(c_ubyte)()
    xlib.XGetWindowProperty(self._x_display, window, atom_property, 0, 2147483647, False, atom_type, byref(actualAtom), byref(actualFormat), byref(itemCount), byref(bytesAfter), data)
    return (data, itemCount.value, actualAtom.value)