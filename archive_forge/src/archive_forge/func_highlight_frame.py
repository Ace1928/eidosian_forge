from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def highlight_frame(frame_id: page.FrameId, content_color: typing.Optional[dom.RGBA]=None, content_outline_color: typing.Optional[dom.RGBA]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Highlights owner element of the frame with given id.

    :param frame_id: Identifier of the frame to highlight.
    :param content_color: *(Optional)* The content box highlight fill color (default: transparent).
    :param content_outline_color: *(Optional)* The content box highlight outline color (default: transparent).
    """
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    if content_color is not None:
        params['contentColor'] = content_color.to_json()
    if content_outline_color is not None:
        params['contentOutlineColor'] = content_outline_color.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.highlightFrame', 'params': params}
    json = (yield cmd_dict)