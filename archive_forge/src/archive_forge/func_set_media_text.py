from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def set_media_text(style_sheet_id: StyleSheetId, range_: SourceRange, text: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, CSSMedia]:
    """
    Modifies the rule selector.

    :param style_sheet_id:
    :param range_:
    :param text:
    :returns: The resulting CSS media rule after modification.
    """
    params: T_JSON_DICT = dict()
    params['styleSheetId'] = style_sheet_id.to_json()
    params['range'] = range_.to_json()
    params['text'] = text
    cmd_dict: T_JSON_DICT = {'method': 'CSS.setMediaText', 'params': params}
    json = (yield cmd_dict)
    return CSSMedia.from_json(json['media'])