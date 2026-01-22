import ctypes
import weakref
from collections import namedtuple
from pyglet.util import debug_print
from pyglet.window.win32 import _user32
from . import lib_dsound as lib
from .exceptions import DirectSoundNativeError
@property
def is_buffer_lost(self):
    return self._get_status() & lib.DSBSTATUS_BUFFERLOST != 0