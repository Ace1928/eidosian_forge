import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def connect_playback(self, tlength: int=_UINT32_MAX, minreq: int=_UINT32_MAX) -> None:
    context = self.context()
    assert self._pa_stream is not None
    assert context is not None
    device = None
    buffer_attr = pa.pa_buffer_attr()
    buffer_attr.fragsize = _UINT32_MAX
    buffer_attr.maxlength = _UINT32_MAX
    buffer_attr.tlength = tlength
    buffer_attr.prebuf = _UINT32_MAX
    buffer_attr.minreq = minreq
    flags = pa.PA_STREAM_START_CORKED | pa.PA_STREAM_INTERPOLATE_TIMING | pa.PA_STREAM_VARIABLE_RATE
    volume = None
    sync_stream = None
    context.check(pa.pa_stream_connect_playback(self._pa_stream, device, buffer_attr, flags, volume, sync_stream))
    while not self.is_ready and (not self.is_failed):
        self.mainloop.wait()
    if not self.is_ready:
        context.raise_error()
    self._refresh_sink_index()
    assert _debug('PulseAudioStream: Playback connected')