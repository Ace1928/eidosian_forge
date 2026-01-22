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
def samples_played(self):
    """Get the amount of samples played by the voice."""
    self._voice.GetState(ctypes.byref(self._voice_state), 0)
    return self._voice_state.SamplesPlayed