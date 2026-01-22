from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def set_attribute_value(node_id: NodeId, name: str, value: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets attribute for an element with given id.

    :param node_id: Id of the element to set attribute for.
    :param name: Attribute name.
    :param value: Attribute value.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['name'] = name
    params['value'] = value
    cmd_dict: T_JSON_DICT = {'method': 'DOM.setAttributeValue', 'params': params}
    json = (yield cmd_dict)