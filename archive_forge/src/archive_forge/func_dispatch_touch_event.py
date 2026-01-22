from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def dispatch_touch_event(type_: str, touch_points: typing.List[TouchPoint], modifiers: typing.Optional[int]=None, timestamp: typing.Optional[TimeSinceEpoch]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Dispatches a touch event to the page.

    :param type_: Type of the touch event. TouchEnd and TouchCancel must not contain any touch points, while TouchStart and TouchMove must contains at least one.
    :param touch_points: Active touch points on the touch device. One event per any changed point (compared to previous touch event in a sequence) is generated, emulating pressing/moving/releasing points one by one.
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param timestamp: *(Optional)* Time at which the event occurred.
    """
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['touchPoints'] = [i.to_json() for i in touch_points]
    if modifiers is not None:
        params['modifiers'] = modifiers
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Input.dispatchTouchEvent', 'params': params}
    json = (yield cmd_dict)