from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def remove_xhr_breakpoint(url: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Removes breakpoint from XMLHttpRequest.

    :param url: Resource URL substring.
    """
    params: T_JSON_DICT = dict()
    params['url'] = url
    cmd_dict: T_JSON_DICT = {'method': 'DOMDebugger.removeXHRBreakpoint', 'params': params}
    json = (yield cmd_dict)