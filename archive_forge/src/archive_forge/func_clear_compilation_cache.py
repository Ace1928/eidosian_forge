from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def clear_compilation_cache() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Clears seeded compilation cache.

    **EXPERIMENTAL**
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.clearCompilationCache'}
    json = (yield cmd_dict)