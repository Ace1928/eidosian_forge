from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def highlight_node() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Highlights DOM node.
    """
    cmd_dict: T_JSON_DICT = {'method': 'DOM.highlightNode'}
    json = (yield cmd_dict)