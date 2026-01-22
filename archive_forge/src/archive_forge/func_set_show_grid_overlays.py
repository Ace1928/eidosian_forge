from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_show_grid_overlays(grid_node_highlight_configs: typing.List[GridNodeHighlightConfig]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Highlight multiple elements with the CSS Grid overlay.

    :param grid_node_highlight_configs: An array of node identifiers and descriptors for the highlight appearance.
    """
    params: T_JSON_DICT = dict()
    params['gridNodeHighlightConfigs'] = [i.to_json() for i in grid_node_highlight_configs]
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setShowGridOverlays', 'params': params}
    json = (yield cmd_dict)