from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_show_paint_rects(result: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Requests that backend shows paint rectangles

    :param result: True for showing paint rectangles
    """
    params: T_JSON_DICT = dict()
    params['result'] = result
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setShowPaintRects', 'params': params}
    json = (yield cmd_dict)