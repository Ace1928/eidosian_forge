from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def request_entries(cache_id: CacheId, skip_count: typing.Optional[int]=None, page_size: typing.Optional[int]=None, path_filter: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.List[DataEntry], float]]:
    """
    Requests data from cache.

    :param cache_id: ID of cache to get entries from.
    :param skip_count: *(Optional)* Number of records to skip.
    :param page_size: *(Optional)* Number of records to fetch.
    :param path_filter: *(Optional)* If present, only return the entries containing this substring in the path
    :returns: A tuple with the following items:

        0. **cacheDataEntries** - Array of object store data entries.
        1. **returnCount** - Count of returned entries from this storage. If pathFilter is empty, it is the count of all entries from this storage.
    """
    params: T_JSON_DICT = dict()
    params['cacheId'] = cache_id.to_json()
    if skip_count is not None:
        params['skipCount'] = skip_count
    if page_size is not None:
        params['pageSize'] = page_size
    if path_filter is not None:
        params['pathFilter'] = path_filter
    cmd_dict: T_JSON_DICT = {'method': 'CacheStorage.requestEntries', 'params': params}
    json = (yield cmd_dict)
    return ([DataEntry.from_json(i) for i in json['cacheDataEntries']], float(json['returnCount']))