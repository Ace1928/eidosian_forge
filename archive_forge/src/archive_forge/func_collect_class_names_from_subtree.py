from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def collect_class_names_from_subtree(node_id: NodeId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[str]]:
    """
    Collects class names for the node with given id and all of it's child nodes.

    **EXPERIMENTAL**

    :param node_id: Id of the node to collect class names.
    :returns: Class name list.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.collectClassNamesFromSubtree', 'params': params}
    json = (yield cmd_dict)
    return [str(i) for i in json['classNames']]