from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def set_style_sheet_text(style_sheet_id: StyleSheetId, text: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Optional[str]]:
    """
    Sets the new stylesheet text.

    :param style_sheet_id:
    :param text:
    :returns: *(Optional)* URL of source map associated with script (if any).
    """
    params: T_JSON_DICT = dict()
    params['styleSheetId'] = style_sheet_id.to_json()
    params['text'] = text
    cmd_dict: T_JSON_DICT = {'method': 'CSS.setStyleSheetText', 'params': params}
    json = (yield cmd_dict)
    return str(json['sourceMapURL']) if 'sourceMapURL' in json else None