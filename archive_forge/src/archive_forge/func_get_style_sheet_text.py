from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def get_style_sheet_text(style_sheet_id: StyleSheetId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    """
    Returns the current textual content for a stylesheet.

    :param style_sheet_id:
    :returns: The stylesheet text.
    """
    params: T_JSON_DICT = dict()
    params['styleSheetId'] = style_sheet_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'CSS.getStyleSheetText', 'params': params}
    json = (yield cmd_dict)
    return str(json['text'])