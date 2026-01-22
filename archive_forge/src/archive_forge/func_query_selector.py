from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def query_selector(node_id: NodeId, selector: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, NodeId]:
    """
    Executes ``querySelector`` on a given node.

    :param node_id: Id of the node to query upon.
    :param selector: Selector string.
    :returns: Query selector result.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    params['selector'] = selector
    cmd_dict: T_JSON_DICT = {'method': 'DOM.querySelector', 'params': params}
    json = (yield cmd_dict)
    return NodeId.from_json(json['nodeId'])