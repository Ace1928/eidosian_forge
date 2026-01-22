from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def get_computed_style_for_node(node_id: dom.NodeId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[CSSComputedStyleProperty]]:
    """
    Returns the computed style for a DOM node identified by ``nodeId``.

    :param node_id:
    :returns: Computed style for the specified DOM node.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'CSS.getComputedStyleForNode', 'params': params}
    json = (yield cmd_dict)
    return [CSSComputedStyleProperty.from_json(i) for i in json['computedStyle']]