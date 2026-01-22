import ctypes
import os
import signal
import struct
import threading
from pyglet.libs.x11 import xlib
from pyglet.util import asbytes
def get_modes_array(display, screen):
    count = ctypes.c_int()
    modes = ctypes.POINTER(ctypes.POINTER(xf86vmode.XF86VidModeModeInfo))()
    xf86vmode.XF86VidModeGetAllModeLines(display, screen, count, modes)
    return (modes, count.value)