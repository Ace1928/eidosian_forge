from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def scroll_into_view_if_needed(node_id: typing.Optional[NodeId]=None, backend_node_id: typing.Optional[BackendNodeId]=None, object_id: typing.Optional[runtime.RemoteObjectId]=None, rect: typing.Optional[Rect]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Scrolls the specified rect of the given node into view if not already visible.
    Note: exactly one between nodeId, backendNodeId and objectId should be passed
    to identify the node.

    :param node_id: *(Optional)* Identifier of the node.
    :param backend_node_id: *(Optional)* Identifier of the backend node.
    :param object_id: *(Optional)* JavaScript object id of the node wrapper.
    :param rect: *(Optional)* The rect to be scrolled into view, relative to the node's border box, in CSS pixels. When omitted, center of the node will be used, similar to Element.scrollIntoView.
    """
    params: T_JSON_DICT = dict()
    if node_id is not None:
        params['nodeId'] = node_id.to_json()
    if backend_node_id is not None:
        params['backendNodeId'] = backend_node_id.to_json()
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if rect is not None:
        params['rect'] = rect.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.scrollIntoViewIfNeeded', 'params': params}
    json = (yield cmd_dict)