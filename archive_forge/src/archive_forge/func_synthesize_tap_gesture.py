from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def synthesize_tap_gesture(x: float, y: float, duration: typing.Optional[int]=None, tap_count: typing.Optional[int]=None, gesture_source_type: typing.Optional[GestureSourceType]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Synthesizes a tap gesture over a time period by issuing appropriate touch events.

    **EXPERIMENTAL**

    :param x: X coordinate of the start of the gesture in CSS pixels.
    :param y: Y coordinate of the start of the gesture in CSS pixels.
    :param duration: *(Optional)* Duration between touchdown and touchup events in ms (default: 50).
    :param tap_count: *(Optional)* Number of times to perform the tap (e.g. 2 for double tap, default: 1).
    :param gesture_source_type: *(Optional)* Which type of input events to be generated (default: 'default', which queries the platform for the preferred input type).
    """
    params: T_JSON_DICT = dict()
    params['x'] = x
    params['y'] = y
    if duration is not None:
        params['duration'] = duration
    if tap_count is not None:
        params['tapCount'] = tap_count
    if gesture_source_type is not None:
        params['gestureSourceType'] = gesture_source_type.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Input.synthesizeTapGesture', 'params': params}
    json = (yield cmd_dict)