import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def update_sample_rate(self, sample_rate: int, callback: Optional[PulseAudioContextSuccessCallback]=None) -> 'PulseAudioOperation':
    context = self.context()
    assert context is not None
    assert self._pa_stream is not None
    clump = PulseAudioStreamSuccessCallbackLump(context, callback)
    return PulseAudioOperation(clump, pa.pa_stream_update_sample_rate(self._pa_stream, sample_rate, clump.pa_callback, None))