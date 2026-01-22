from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def get_dom_counters() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[int, int, int]]:
    """


    :returns: A tuple with the following items:

        0. **documents** - 
        1. **nodes** - 
        2. **jsEventListeners** - 
    """
    cmd_dict: T_JSON_DICT = {'method': 'Memory.getDOMCounters'}
    json = (yield cmd_dict)
    return (int(json['documents']), int(json['nodes']), int(json['jsEventListeners']))