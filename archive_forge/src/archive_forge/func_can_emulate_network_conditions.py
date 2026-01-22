from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
def can_emulate_network_conditions() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, bool]:
    """
    Tells whether emulation of network conditions is supported.

    :returns: True if emulation of network conditions is supported.
    """
    cmd_dict: T_JSON_DICT = {'method': 'Network.canEmulateNetworkConditions'}
    json = (yield cmd_dict)
    return bool(json['result'])