from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def track_computed_style_updates(properties_to_track: typing.List[CSSComputedStyleProperty]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Starts tracking the given computed styles for updates. The specified array of properties
    replaces the one previously specified. Pass empty array to disable tracking.
    Use takeComputedStyleUpdates to retrieve the list of nodes that had properties modified.
    The changes to computed style properties are only tracked for nodes pushed to the front-end
    by the DOM agent. If no changes to the tracked properties occur after the node has been pushed
    to the front-end, no updates will be issued for the node.

    **EXPERIMENTAL**

    :param properties_to_track:
    """
    params: T_JSON_DICT = dict()
    params['propertiesToTrack'] = [i.to_json() for i in properties_to_track]
    cmd_dict: T_JSON_DICT = {'method': 'CSS.trackComputedStyleUpdates', 'params': params}
    json = (yield cmd_dict)