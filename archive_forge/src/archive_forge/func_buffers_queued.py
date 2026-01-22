from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
@property
def buffers_queued(self):
    """Get the amount of buffers in the current voice. Adding flag for no samples played is 3x faster."""
    self._voice.GetState(ctypes.byref(self._voice_state), lib.XAUDIO2_VOICE_NOSAMPLESPLAYED)
    return self._voice_state.BuffersQueued