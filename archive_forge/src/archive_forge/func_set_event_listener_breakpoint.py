from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def set_event_listener_breakpoint(event_name: str, target_name: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets breakpoint on particular DOM event.

    :param event_name: DOM Event name to stop on (any DOM event will do).
    :param target_name: **(EXPERIMENTAL)** *(Optional)* EventTarget interface name to stop on. If equal to ```"*"``` or not provided, will stop on any EventTarget.
    """
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    if target_name is not None:
        params['targetName'] = target_name
    cmd_dict: T_JSON_DICT = {'method': 'DOMDebugger.setEventListenerBreakpoint', 'params': params}
    json = (yield cmd_dict)