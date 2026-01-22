from __future__ import annotations
from typing import Any
from unittest import mock
import pytest
from xarray.backends.lru_cache import LRUCache
def test_update_priority() -> None:
    cache: LRUCache[Any, Any] = LRUCache(maxsize=2)
    cache['x'] = 1
    cache['y'] = 2
    assert list(cache) == ['x', 'y']
    assert 'x' in cache
    assert list(cache) == ['y', 'x']
    assert cache['y'] == 2
    assert list(cache) == ['x', 'y']
    cache['x'] = 3
    assert list(cache.items()) == [('y', 2), ('x', 3)]