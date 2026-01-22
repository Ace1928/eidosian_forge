import ctypes
import weakref
from collections import namedtuple
from pyglet.util import debug_print
from pyglet.window.win32 import _user32
from . import lib_dsound as lib
from .exceptions import DirectSoundNativeError
def create_buffer(self, audio_format, buffer_size):
    wave_format = _create_wave_format(audio_format)
    buffer_desc = _create_buffer_desc(wave_format, buffer_size)
    return DirectSoundBuffer(self._create_native_buffer(buffer_desc), audio_format, buffer_size)