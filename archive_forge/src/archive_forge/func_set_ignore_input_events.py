from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_ignore_input_events(ignore: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Ignores input events (useful while auditing page).

    :param ignore: Ignores input events processing when set to true.
    """
    params: T_JSON_DICT = dict()
    params['ignore'] = ignore
    cmd_dict: T_JSON_DICT = {'method': 'Input.setIgnoreInputEvents', 'params': params}
    json = (yield cmd_dict)