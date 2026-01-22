from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def push_nodes_by_backend_ids_to_frontend(backend_node_ids: typing.List[BackendNodeId]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[NodeId]]:
    """
    Requests that a batch of nodes is sent to the caller given their backend node ids.

    **EXPERIMENTAL**

    :param backend_node_ids: The array of backend node ids.
    :returns: The array of ids of pushed nodes that correspond to the backend ids specified in backendNodeIds.
    """
    params: T_JSON_DICT = dict()
    params['backendNodeIds'] = [i.to_json() for i in backend_node_ids]
    cmd_dict: T_JSON_DICT = {'method': 'DOM.pushNodesByBackendIdsToFrontend', 'params': params}
    json = (yield cmd_dict)
    return [NodeId.from_json(i) for i in json['nodeIds']]