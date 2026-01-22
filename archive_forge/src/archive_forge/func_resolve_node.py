from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def resolve_node(node_id: typing.Optional[NodeId]=None, backend_node_id: typing.Optional[BackendNodeId]=None, object_group: typing.Optional[str]=None, execution_context_id: typing.Optional[runtime.ExecutionContextId]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, runtime.RemoteObject]:
    """
    Resolves the JavaScript node object for a given NodeId or BackendNodeId.

    :param node_id: *(Optional)* Id of the node to resolve.
    :param backend_node_id: *(Optional)* Backend identifier of the node to resolve.
    :param object_group: *(Optional)* Symbolic group name that can be used to release multiple objects.
    :param execution_context_id: *(Optional)* Execution context in which to resolve the node.
    :returns: JavaScript object wrapper for given node.
    """
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_group is not None:
        params['objectGroup'] = object_group
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.resolveNode', 'params': params}
    json = (yield cmd_dict)
    return runtime.RemoteObject.from_json(json['object'])