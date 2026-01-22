from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def set_inspected_node(node_id: NodeId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enables console to refer to the node with given id via $x (see Command Line API for more details
    $x functions).

    **EXPERIMENTAL**

    :param node_id: DOM node id to be accessible by means of $x command line API.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.setInspectedNode', 'params': params}
    json = (yield cmd_dict)