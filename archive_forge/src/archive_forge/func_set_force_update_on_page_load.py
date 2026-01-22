from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
def set_force_update_on_page_load(force_update_on_page_load: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param force_update_on_page_load:
    """
    params: T_JSON_DICT = dict()
    params['forceUpdateOnPageLoad'] = force_update_on_page_load
    cmd_dict: T_JSON_DICT = {'method': 'ServiceWorker.setForceUpdateOnPageLoad', 'params': params}
    json = (yield cmd_dict)