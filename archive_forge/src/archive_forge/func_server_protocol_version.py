import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
@property
def server_protocol_version(self) -> Optional[str]:
    if self._pa_context is not None:
        return get_uint32_or_none(pa.pa_context_get_server_protocol_version(self._pa_context))
    return None