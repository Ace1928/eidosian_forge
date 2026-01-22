import queue
import atexit
import weakref
import tempfile
from threading import Event, Thread
from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder
def new_sample(self, sink):
    """new-sample callback"""
    buffer = sink.emit('pull-sample').get_buffer()
    mem = buffer.extract_dup(0, buffer.get_size())
    self.source.queue.put(mem)
    return Gst.FlowReturn.OK