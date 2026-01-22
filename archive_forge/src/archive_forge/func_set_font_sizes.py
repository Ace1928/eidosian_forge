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
def set_font_sizes(font_sizes: FontSizes) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Set default font sizes.

    **EXPERIMENTAL**

    :param font_sizes: Specifies font sizes to set. If a font size is not specified, it won't be changed.
    """
    params: T_JSON_DICT = dict()
    params['fontSizes'] = font_sizes.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.setFontSizes', 'params': params}
    json = (yield cmd_dict)