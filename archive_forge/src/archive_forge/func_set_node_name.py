from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def set_node_name(node_id: NodeId, name: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, NodeId]:
    """
    Sets node name for a node with given id.

    :param node_id: Id of the node to set name for.
    :param name: New node's name.
    :returns: New node's id.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['name'] = name
    cmd_dict: T_JSON_DICT = {'method': 'DOM.setNodeName', 'params': params}
    json = (yield cmd_dict)
    return NodeId.from_json(json['nodeId'])