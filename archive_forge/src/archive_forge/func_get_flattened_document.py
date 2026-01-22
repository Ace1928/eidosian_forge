from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
def get_flattened_document(depth: typing.Optional[int]=None, pierce: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[Node]]:
    """
    Returns the root DOM node (and optionally the subtree) to the caller.
    Deprecated, as it is not designed to work well with the rest of the DOM agent.
    Use DOMSnapshot.captureSnapshot instead.

    :param depth: *(Optional)* The maximum depth at which children should be retrieved, defaults to 1. Use -1 for the entire subtree or provide an integer larger than 0.
    :param pierce: *(Optional)* Whether or not iframes and shadow roots should be traversed when returning the subtree (default is false).
    :returns: Resulting node.
    """
    params: T_JSON_DICT = dict()
    if depth is not None:
        params['depth'] = depth
    if pierce is not None:
        params['pierce'] = pierce
    cmd_dict: T_JSON_DICT = {'method': 'DOM.getFlattenedDocument', 'params': params}
    json = (yield cmd_dict)
    return [Node.from_json(i) for i in json['nodes']]