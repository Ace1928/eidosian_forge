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
def get_resource_tree() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, FrameResourceTree]:
    """
    Returns present frame / resource tree structure.

    **EXPERIMENTAL**

    :returns: Present frame / resource tree structure.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Page.getResourceTree'}
    json = (yield cmd_dict)
    return FrameResourceTree.from_json(json['frameTree'])