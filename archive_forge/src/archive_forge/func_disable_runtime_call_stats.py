from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
def disable_runtime_call_stats() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Disable run time call stats collection.

    **EXPERIMENTAL**
    """
    cmd_dict: T_JSON_DICT = {'method': 'Profiler.disableRuntimeCallStats'}
    json = (yield cmd_dict)