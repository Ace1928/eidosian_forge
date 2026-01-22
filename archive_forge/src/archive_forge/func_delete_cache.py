from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def delete_cache(cache_id: CacheId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Deletes a cache.

    :param cache_id: Id of cache for deletion.
    """
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'CacheStorage.deleteCache', 'params': params}
    json = (yield cmd_dict)