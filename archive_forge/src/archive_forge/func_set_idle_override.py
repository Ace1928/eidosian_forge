from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_idle_override(is_user_active: bool, is_screen_unlocked: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Overrides the Idle state.

    :param is_user_active: Mock isUserActive
    :param is_screen_unlocked: Mock isScreenUnlocked
    """
    params: T_JSON_DICT = dict()
    params['isUserActive'] = is_user_active
    params['isScreenUnlocked'] = is_screen_unlocked
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setIdleOverride', 'params': params}
    json = (yield cmd_dict)