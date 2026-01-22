from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def set_local_fonts_enabled(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enables/disables rendering of local CSS fonts (enabled by default).

    **EXPERIMENTAL**

    :param enabled: Whether rendering of local fonts is enabled.
    """
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'CSS.setLocalFontsEnabled', 'params': params}
    json = (yield cmd_dict)