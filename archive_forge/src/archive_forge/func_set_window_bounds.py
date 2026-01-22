from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
def set_window_bounds(window_id: WindowID, bounds: Bounds) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Set position and/or size of the browser window.

    **EXPERIMENTAL**

    :param window_id: Browser window id.
    :param bounds: New window bounds. The 'minimized', 'maximized' and 'fullscreen' states cannot be combined with 'left', 'top', 'width' or 'height'. Leaves unspecified fields unchanged.
    """
    params: T_JSON_DICT = dict()
    params['windowId'] = window_id.to_json()
    params['bounds'] = bounds.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Browser.setWindowBounds', 'params': params}
    json = (yield cmd_dict)