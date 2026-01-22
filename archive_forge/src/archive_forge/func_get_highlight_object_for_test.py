from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def get_highlight_object_for_test(node_id: dom.NodeId, include_distance: typing.Optional[bool]=None, include_style: typing.Optional[bool]=None, color_format: typing.Optional[ColorFormat]=None, show_accessibility_info: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, dict]:
    """
    For testing.

    :param node_id: Id of the node to get highlight object for.
    :param include_distance: *(Optional)* Whether to include distance info.
    :param include_style: *(Optional)* Whether to include style info.
    :param color_format: *(Optional)* The color format to get config with (default: hex).
    :param show_accessibility_info: *(Optional)* Whether to show accessibility info (default: true).
    :returns: Highlight data for the node.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    if include_distance is not None:
        params['includeDistance'] = include_distance
    if include_style is not None:
        params['includeStyle'] = include_style
    if color_format is not None:
        params['colorFormat'] = color_format.to_json()
    if show_accessibility_info is not None:
        params['showAccessibilityInfo'] = show_accessibility_info
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.getHighlightObjectForTest', 'params': params}
    json = (yield cmd_dict)
    return dict(json['highlight'])