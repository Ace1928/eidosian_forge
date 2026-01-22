from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def get_all_time_sampling_profile() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, SamplingProfile]:
    """
    Retrieve native memory allocations profile
    collected since renderer process startup.

    :returns: 
    """
    cmd_dict: T_JSON_DICT = {'method': 'Memory.getAllTimeSamplingProfile'}
    json = (yield cmd_dict)
    return SamplingProfile.from_json(json['profile'])