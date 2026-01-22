from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def get_partial_ax_tree(node_id: typing.Optional[dom.NodeId]=None, backend_node_id: typing.Optional[dom.BackendNodeId]=None, object_id: typing.Optional[runtime.RemoteObjectId]=None, fetch_relatives: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[AXNode]]:
    """
    Fetches the accessibility node and partial accessibility tree for this DOM node, if it exists.

    **EXPERIMENTAL**

    :param node_id: *(Optional)* Identifier of the node to get the partial accessibility tree for.
    :param backend_node_id: *(Optional)* Identifier of the backend node to get the partial accessibility tree for.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper to get the partial accessibility tree for.
    :param fetch_relatives: *(Optional)* Whether to fetch this nodes ancestors, siblings and children. Defaults to true.
    :returns: The ``Accessibility.AXNode`` for this DOM node, if it exists, plus its ancestors, siblings and children, if requested.
    """
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if fetch_relatives is not None:
        params['fetchRelatives'] = fetch_relatives
    cmd_dict: T_JSON_DICT = {'method': 'Accessibility.getPartialAXTree', 'params': params}
    json = (yield cmd_dict)
    return [AXNode.from_json(i) for i in json['nodes']]