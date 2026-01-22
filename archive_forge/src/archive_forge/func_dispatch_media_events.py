from collections import deque
import ctypes
import weakref
from abc import ABCMeta, abstractmethod
import pyglet
from pyglet.media.codecs import AudioData
from pyglet.util import debug_print
def dispatch_media_events(self, until_cursor):
    """Dispatch all :class:`MediaEvent`s whose index is less than or equal
        to the specified ``until_cursor`` (which should be a very recent play
        cursor position).
        Please note that :attr:`_compensated_bytes` will be subtracted from
        the passed ``until_cursor``.
        """
    until_cursor = self._to_perceived_play_cursor(until_cursor)
    while self._events and self._events[0][0] <= until_cursor:
        self._events.popleft()[1].sync_dispatch_to_player(self.player)