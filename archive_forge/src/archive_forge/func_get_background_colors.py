from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def get_background_colors(node_id: dom.NodeId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.Optional[typing.List[str]], typing.Optional[str], typing.Optional[str]]]:
    """
    :param node_id: Id of the node to get background colors for.
    :returns: A tuple with the following items:

        0. **backgroundColors** - *(Optional)* The range of background colors behind this element, if it contains any visible text. If no visible text is present, this will be undefined. In the case of a flat background color, this will consist of simply that color. In the case of a gradient, this will consist of each of the color stops. For anything more complicated, this will be an empty array. Images will be ignored (as if the image had failed to load).
        1. **computedFontSize** - *(Optional)* The computed font size for this node, as a CSS computed value string (e.g. '12px').
        2. **computedFontWeight** - *(Optional)* The computed font weight for this node, as a CSS computed value string (e.g. 'normal' or '100').
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'CSS.getBackgroundColors', 'params': params}
    json = (yield cmd_dict)
    return ([str(i) for i in json['backgroundColors']] if 'backgroundColors' in json else None, str(json['computedFontSize']) if 'computedFontSize' in json else None, str(json['computedFontWeight']) if 'computedFontWeight' in json else None)