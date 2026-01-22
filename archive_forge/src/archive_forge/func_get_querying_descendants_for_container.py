from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def get_querying_descendants_for_container(node_id: NodeId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[NodeId]]:
    """
    Returns the descendants of a container query container that have
    container queries against this container.

    **EXPERIMENTAL**

    :param node_id: Id of the container node to find querying descendants from.
    :returns: Descendant nodes with container queries against the given container.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.getQueryingDescendantsForContainer', 'params': params}
    json = (yield cmd_dict)
    return [NodeId.from_json(i) for i in json['nodeIds']]