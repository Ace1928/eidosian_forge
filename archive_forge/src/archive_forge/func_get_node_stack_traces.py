from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def get_node_stack_traces(node_id: NodeId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Optional[runtime.StackTrace]]:
    """
    Gets stack traces associated with a Node. As of now, only provides stack trace for Node creation.

    **EXPERIMENTAL**

    :param node_id: Id of the node to get stack traces for.
    :returns: *(Optional)* Creation stack trace, if available.
    """
    params: T_JSON_DICT = dict()
    params['nodeId'] = node_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'DOM.getNodeStackTraces', 'params': params}
    json = (yield cmd_dict)
    return runtime.StackTrace.from_json(json['creation']) if 'creation' in json else None