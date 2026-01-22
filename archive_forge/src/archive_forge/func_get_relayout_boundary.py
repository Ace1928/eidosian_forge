from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def get_relayout_boundary(node_id: NodeId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, NodeId]:
    """
    Returns the id of the nearest ancestor that is a relayout boundary.

    **EXPERIMENTAL**

    :param node_id: Id of the node.
    :returns: Relayout boundary node id for the given node.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.getRelayoutBoundary', 'params': params}
    json = (yield cmd_dict)
    return NodeId.from_json(json['nodeId'])