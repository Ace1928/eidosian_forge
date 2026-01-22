from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def mark_undoable_state() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Marks last undoable state.

    **EXPERIMENTAL**
    """
    cmd_dict: T_JSON_DICT = {'method': 'DOM.markUndoableState'}
    json = (yield cmd_dict)