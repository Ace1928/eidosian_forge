from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import runtime
def take_type_profile() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[ScriptTypeProfile]]:
    """
    Collect type profile.

    **EXPERIMENTAL**

    :returns: Type profile for all scripts since startTypeProfile() was turned on.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Profiler.takeTypeProfile'}
    json = (yield cmd_dict)
    return [ScriptTypeProfile.from_json(i) for i in json['result']]