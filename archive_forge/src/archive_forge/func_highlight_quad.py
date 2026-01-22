from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def highlight_quad(quad: dom.Quad, color: typing.Optional[dom.RGBA]=None, outline_color: typing.Optional[dom.RGBA]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Highlights given quad. Coordinates are absolute with respect to the main frame viewport.

    :param quad: Quad to highlight
    :param color: *(Optional)* The highlight fill color (default: transparent).
    :param outline_color: *(Optional)* The highlight outline color (default: transparent).
    """
    params: T_JSON_DICT = dict()
    params['quad'] = quad.to_json()
    if color is not None:
        params['color'] = color.to_json()
    if outline_color is not None:
        params['outlineColor'] = outline_color.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.highlightQuad', 'params': params}
    json = (yield cmd_dict)