import queue
import atexit
import weakref
import tempfile
from threading import Event, Thread
from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder
def notify_caps(self, pad, *args):
    """notify::caps callback"""
    self.source.caps = True
    info = pad.get_current_caps().get_structure(0)
    self.source._duration = pad.get_peer().query_duration(Gst.Format.TIME).duration / Gst.SECOND
    channels = info.get_int('channels')[1]
    sample_rate = info.get_int('rate')[1]
    sample_size = int(''.join(filter(str.isdigit, info.get_string('format'))))
    self.source.audio_format = AudioFormat(channels=channels, sample_size=sample_size, sample_rate=sample_rate)
    self.source.is_ready.set()