from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def set_property_rule_property_name(style_sheet_id: StyleSheetId, range_: SourceRange, property_name: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, Value]:
    """
    Modifies the property rule property name.

    :param style_sheet_id:
    :param range_:
    :param property_name:
    :returns: The resulting key text after modification.
    """
    params: T_JSON_DICT = dict()
    params['styleSheetId'] = style_sheet_id.to_json()
    params['range'] = range_.to_json()
    params['propertyName'] = property_name
    cmd_dict: T_JSON_DICT = {'method': 'CSS.setPropertyRulePropertyName', 'params': params}
    json = (yield cmd_dict)
    return Value.from_json(json['propertyName'])