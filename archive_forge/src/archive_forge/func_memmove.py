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
def memmove(self, target_pointer: int, num_bytes: int) -> int:
    bytes_written = 0
    bytes_remaining = num_bytes
    while bytes_remaining > 0 and self._data:
        cur_audio_data = self._data[0]
        cur_len = cur_audio_data.length - self._first_read_offset
        packet_used = cur_len <= bytes_remaining
        cur_write = min(bytes_remaining, cur_len)
        ctypes.memmove(target_pointer + bytes_written, cur_audio_data.pointer + self._first_read_offset, cur_write)
        bytes_written += cur_write
        bytes_remaining -= cur_write
        if packet_used:
            self._data.popleft()
            self._first_read_offset = 0
        else:
            self._first_read_offset += cur_write
    self.available -= bytes_written
    return bytes_written