from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def set_playback_rate(playback_rate: float) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets the playback rate of the document timeline.

    :param playback_rate: Playback rate for animations on page
    """
    params: T_JSON_DICT = dict()
    params['playbackRate'] = playback_rate
    cmd_dict: T_JSON_DICT = {'method': 'Animation.setPlaybackRate', 'params': params}
    json = (yield cmd_dict)