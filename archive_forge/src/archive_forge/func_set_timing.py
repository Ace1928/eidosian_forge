from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def set_timing(animation_id: str, duration: float, delay: float) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets the timing of an animation node.

    :param animation_id: Animation id.
    :param duration: Duration of the animation.
    :param delay: Delay of the animation.
    """
    params: T_JSON_DICT = dict()
    params['animationId'] = animation_id
    params['duration'] = duration
    params['delay'] = delay
    cmd_dict: T_JSON_DICT = {'method': 'Animation.setTiming', 'params': params}
    json = (yield cmd_dict)