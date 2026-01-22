from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def untrack_cache_storage_for_storage_key(storage_key: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Unregisters storage key from receiving notifications for cache storage.

    :param storage_key: Storage key.
    """
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    cmd_dict: T_JSON_DICT = {'method': 'Storage.untrackCacheStorageForStorageKey', 'params': params}
    json = (yield cmd_dict)