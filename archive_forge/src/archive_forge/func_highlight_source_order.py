from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def highlight_source_order(source_order_config: SourceOrderConfig, node_id: typing.Optional[dom.NodeId]=None, backend_node_id: typing.Optional[dom.BackendNodeId]=None, object_id: typing.Optional[runtime.RemoteObjectId]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Highlights the source order of the children of the DOM node with given id or with the given
    JavaScript object wrapper. Either nodeId or objectId must be specified.

    :param source_order_config: A descriptor for the appearance of the overlay drawing.
    :param node_id: *(Optional)* Identifier of the node to highlight.
    :param backend_node_id: *(Optional)* Identifier of the backend node to highlight.
    :param object_id: *(Optional)* JavaScript object id of the node to be highlighted.
    """
    params: T_JSON_DICT = dict()
    params['sourceOrderConfig'] = source_order_config.to_json()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.highlightSourceOrder', 'params': params}
    json = (yield cmd_dict)