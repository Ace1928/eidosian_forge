from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def get_platform_fonts_for_node(node_id: dom.NodeId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[PlatformFontUsage]]:
    """
    Requests information about platform fonts which we used to render child TextNodes in the given
    node.

    :param node_id:
    :returns: Usage statistics for every employed platform font.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'CSS.getPlatformFontsForNode', 'params': params}
    json = (yield cmd_dict)
    return [PlatformFontUsage.from_json(i) for i in json['fonts']]