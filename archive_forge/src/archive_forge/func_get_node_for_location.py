from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def get_node_for_location(x: int, y: int, include_user_agent_shadow_dom: typing.Optional[bool]=None, ignore_pointer_events_none: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[BackendNodeId, page.FrameId, typing.Optional[NodeId]]]:
    """
    Returns node id at given location. Depending on whether DOM domain is enabled, nodeId is
    either returned or not.

    :param x: X coordinate.
    :param y: Y coordinate.
    :param include_user_agent_shadow_dom: *(Optional)* False to skip to the nearest non-UA shadow root ancestor (default: false).
    :param ignore_pointer_events_none: *(Optional)* Whether to ignore pointer-events: none on elements and hit test them.
    :returns: A tuple with the following items:

        0. **backendNodeId** - Resulting node.
        1. **frameId** - Frame this node belongs to.
        2. **nodeId** - *(Optional)* Id of the node at given coordinates, only when enabled and requested document.
    """
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    if include_user_agent_shadow_dom is not None:
        params['includeUserAgentShadowDOM'] = include_user_agent_shadow_dom
    if ignore_pointer_events_none is not None:
        params['ignorePointerEventsNone'] = ignore_pointer_events_none
    cmd_dict: T_JSON_DICT = {'method': 'DOM.getNodeForLocation', 'params': params}
    json = (yield cmd_dict)
    return (BackendNodeId.from_json(json['backendNodeId']), page.FrameId.from_json(json['frameId']), NodeId.from_json(json['nodeId']) if 'nodeId' in json else None)