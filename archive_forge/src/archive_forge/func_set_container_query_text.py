from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def set_container_query_text(style_sheet_id: StyleSheetId, range_: SourceRange, text: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, CSSContainerQuery]:
    """
    Modifies the expression of a container query.

    **EXPERIMENTAL**

    :param style_sheet_id:
    :param range_:
    :param text:
    :returns: The resulting CSS container query rule after modification.
    """
    params: T_JSON_DICT = dict()
    params['styleSheetId'] = style_sheet_id.to_json()
    params['range'] = range_.to_json()
    params['text'] = text
    cmd_dict: T_JSON_DICT = {'method': 'CSS.setContainerQueryText', 'params': params}
    json = (yield cmd_dict)
    return CSSContainerQuery.from_json(json['containerQuery'])