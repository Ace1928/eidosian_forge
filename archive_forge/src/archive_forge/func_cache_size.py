from typing import Callable, Dict, Generic, Iterable, Iterator, Optional, TypeVar
def cache_size(self) -> int:
    """Get the number of entries we will cache."""
    return self._max_cache