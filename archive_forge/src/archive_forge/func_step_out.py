from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def step_out() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Steps out of the function call.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.stepOut'}
    json = (yield cmd_dict)