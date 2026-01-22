from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def step_over() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Steps over the statement.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.stepOver'}
    json = (yield cmd_dict)