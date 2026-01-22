from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def dispatch_mouse_event(type_: str, x: float, y: float, modifiers: typing.Optional[int]=None, timestamp: typing.Optional[TimeSinceEpoch]=None, button: typing.Optional[MouseButton]=None, buttons: typing.Optional[int]=None, click_count: typing.Optional[int]=None, delta_x: typing.Optional[float]=None, delta_y: typing.Optional[float]=None, pointer_type: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Dispatches a mouse event to the page.

    :param type_: Type of the mouse event.
    :param x: X coordinate of the event relative to the main frame's viewport in CSS pixels.
    :param y: Y coordinate of the event relative to the main frame's viewport in CSS pixels. 0 refers to the top of the viewport and Y increases as it proceeds towards the bottom of the viewport.
    :param modifiers: *(Optional)* Bit field representing pressed modifier keys. Alt=1, Ctrl=2, Meta/Command=4, Shift=8 (default: 0).
    :param timestamp: *(Optional)* Time at which the event occurred.
    :param button: *(Optional)* Mouse button (default: "none").
    :param buttons: *(Optional)* A number indicating which buttons are pressed on the mouse when a mouse event is triggered. Left=1, Right=2, Middle=4, Back=8, Forward=16, None=0.
    :param click_count: *(Optional)* Number of times the mouse button was clicked (default: 0).
    :param delta_x: *(Optional)* X delta in CSS pixels for mouse wheel event (default: 0).
    :param delta_y: *(Optional)* Y delta in CSS pixels for mouse wheel event (default: 0).
    :param pointer_type: *(Optional)* Pointer type (default: "mouse").
    """
    params: T_JSON_DICT = dict()
    params['type'] = type_
    params['x'] = x
    params['y'] = y
    if modifiers is not None:
        params['modifiers'] = modifiers
    if timestamp is not None:
        params['timestamp'] = timestamp.to_json()
    if button is not None:
        params['button'] = button.to_json()
    if buttons is not None:
        params['buttons'] = buttons
    if click_count is not None:
        params['clickCount'] = click_count
    if delta_x is not None:
        params['deltaX'] = delta_x
    if delta_y is not None:
        params['deltaY'] = delta_y
    if pointer_type is not None:
        params['pointerType'] = pointer_type
    cmd_dict: T_JSON_DICT = {'method': 'Input.dispatchMouseEvent', 'params': params}
    json = (yield cmd_dict)