from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def clear_idle_override() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Clears Idle state overrides.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.clearIdleOverride'}
    json = (yield cmd_dict)