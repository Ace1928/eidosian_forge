from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def request_node(object_id: runtime.RemoteObjectId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, NodeId]:
    """
    Requests that the node is sent to the caller given the JavaScript node object reference. All
    nodes that form the path from the node to the root are also sent to the client as a series of
    ``setChildNodes`` notifications.

    :param object_id: JavaScript object id to convert into node.
    :returns: Node id for given object.
    """
    params: T_JSON_DICT = dict()
    params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.requestNode', 'params': params}
    json = (yield cmd_dict)
    return NodeId.from_json(json['nodeId'])