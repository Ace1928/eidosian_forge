from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def collect_garbage() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    cmd_dict: T_JSON_DICT = {'method': 'HeapProfiler.collectGarbage'}
    json = (yield cmd_dict)