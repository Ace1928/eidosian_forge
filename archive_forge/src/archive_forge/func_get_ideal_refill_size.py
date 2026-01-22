from collections import deque
import ctypes
import threading
from typing import Deque, Optional, TYPE_CHECKING
import weakref
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
from . import lib_pulseaudio as pa
from .interface import PulseAudioMainloop
def get_ideal_refill_size(self, virtual_required: int=0) -> int:
    virtual_available = self.available - virtual_required
    if virtual_available < self._comfortable_limit:
        return self._ideal_size - virtual_available
    return 0