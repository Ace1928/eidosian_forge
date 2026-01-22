from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def get_grid_highlight_objects_for_test(node_ids: typing.List[dom.NodeId]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, dict]:
    """
    For Persistent Grid testing.

    :param node_ids: Ids of the node to get highlight object for.
    :returns: Grid Highlight data for the node ids provided.
    """
    params: T_JSON_DICT = dict()
    params['nodeIds'] = [i.to_json() for i in node_ids]
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.getGridHighlightObjectsForTest', 'params': params}
    json = (yield cmd_dict)
    return dict(json['highlights'])