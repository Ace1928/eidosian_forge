from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def release_animations(animations: typing.List[str]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Releases a set of animations to no longer be manipulated.

    :param animations: List of animation ids to seek.
    """
    params: T_JSON_DICT = dict()
    params['animations'] = [i for i in animations]
    cmd_dict: T_JSON_DICT = {'method': 'Animation.releaseAnimations', 'params': params}
    json = (yield cmd_dict)