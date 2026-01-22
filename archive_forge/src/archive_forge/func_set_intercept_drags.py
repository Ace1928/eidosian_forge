from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_intercept_drags(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Prevents default drag and drop behavior and instead emits ``Input.dragIntercepted`` events.
    Drag and drop behavior can be directly controlled via ``Input.dispatchDragEvent``.

    **EXPERIMENTAL**

    :param enabled:
    """
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Input.setInterceptDrags', 'params': params}
    json = (yield cmd_dict)