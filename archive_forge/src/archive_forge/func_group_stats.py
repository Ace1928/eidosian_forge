from __future__ import annotations
import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable
def group_stats(stats: list[CacheStat]) -> list[CacheStat]:
    """Group a list of CacheStats by category_name and cache_name and sum byte_length"""

    def key_function(individual_stat):
        return (individual_stat.category_name, individual_stat.cache_name)
    result: list[CacheStat] = []
    sorted_stats = sorted(stats, key=key_function)
    grouped_stats = itertools.groupby(sorted_stats, key=key_function)
    for (category_name, cache_name), single_group_stats in grouped_stats:
        result.append(CacheStat(category_name=category_name, cache_name=cache_name, byte_length=sum(map(lambda item: item.byte_length, single_group_stats))))
    return result