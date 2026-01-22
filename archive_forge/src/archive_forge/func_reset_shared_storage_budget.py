from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def reset_shared_storage_budget(owner_origin: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Resets the budget for ``ownerOrigin`` by clearing all budget withdrawals.

    **EXPERIMENTAL**

    :param owner_origin:
    """
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    cmd_dict: T_JSON_DICT = {'method': 'Storage.resetSharedStorageBudget', 'params': params}
    json = (yield cmd_dict)