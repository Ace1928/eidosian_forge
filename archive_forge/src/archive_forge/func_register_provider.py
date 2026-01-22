from __future__ import annotations
import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable
def register_provider(self, provider: CacheStatsProvider) -> None:
    """Register a CacheStatsProvider with the manager.
        This function is not thread-safe. Call it immediately after
        creation.
        """
    self._cache_stats_providers.append(provider)