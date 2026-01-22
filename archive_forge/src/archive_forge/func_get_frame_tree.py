from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def get_frame_tree() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, FrameTree]:
    """
    Returns present frame tree structure.

    :returns: Present frame tree structure.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.getFrameTree'}
    json = (yield cmd_dict)
    return FrameTree.from_json(json['frameTree'])