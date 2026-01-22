from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def get_sampling_profile() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, SamplingHeapProfile]:
    """


    :returns: Return the sampling profile being collected.
    """
    cmd_dict: T_JSON_DICT = {'method': 'HeapProfiler.getSamplingProfile'}
    json = (yield cmd_dict)
    return SamplingHeapProfile.from_json(json['profile'])