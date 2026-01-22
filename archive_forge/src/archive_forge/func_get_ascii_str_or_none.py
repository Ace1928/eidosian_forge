import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
def get_ascii_str_or_none(value: Optional[bytes]) -> Optional[str]:
    if value is not None:
        return value.decode('ascii')
    return None