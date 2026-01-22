from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def set_rule_selector(style_sheet_id: StyleSheetId, range_: SourceRange, selector: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, SelectorList]:
    """
    Modifies the rule selector.

    :param style_sheet_id:
    :param range_:
    :param selector:
    :returns: The resulting selector list after modification.
    """
    params: T_JSON_DICT = dict()
    params['styleSheetId'] = style_sheet_id.to_json()
    params['range'] = range_.to_json()
    params['selector'] = selector
    cmd_dict: T_JSON_DICT = {'method': 'CSS.setRuleSelector', 'params': params}
    json = (yield cmd_dict)
    return SelectorList.from_json(json['selectorList'])