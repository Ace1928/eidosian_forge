from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def set_instrumentation_breakpoint(event_name: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets breakpoint on particular native event.

    **EXPERIMENTAL**

    :param event_name: Instrumentation name to stop on.
    """
    params: T_JSON_DICT = dict()
    params['eventName'] = event_name
    cmd_dict: T_JSON_DICT = {'method': 'DOMDebugger.setInstrumentationBreakpoint', 'params': params}
    json = (yield cmd_dict)