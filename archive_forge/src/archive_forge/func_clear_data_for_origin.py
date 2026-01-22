from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def clear_data_for_origin(origin: str, storage_types: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Clears storage for origin.

    :param origin: Security origin.
    :param storage_types: Comma separated list of StorageType to clear.
    """
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['storageTypes'] = storage_types
    cmd_dict: T_JSON_DICT = {'method': 'Storage.clearDataForOrigin', 'params': params}
    json = (yield cmd_dict)