import ctypes
import weakref
from collections import namedtuple
from . import lib_openal as al
from . import lib_alc as alc
from pyglet.util import debug_print
from pyglet.media.exceptions import MediaException
def queue_buffer(self, buf):
    assert buf.is_valid
    al.alSourceQueueBuffers(self._al_source, 1, ctypes.byref(buf.al_name))
    self._check_error('Failed to queue buffer.')
    self._owned_buffers[buf.name] = buf