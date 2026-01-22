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
def cone_angles(self):
    """The inside and outside angles of the sound projection cone."""
    if self.is_emitter:
        return self._ConeAngles(self._emitter.pCone.contents.InnerAngle, self._emitter.pCone.contents.OuterAngle)
    else:
        return self._ConeAngles(0, 0)