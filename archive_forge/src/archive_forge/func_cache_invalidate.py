import asyncio
import dataclasses
import sys
from asyncio.coroutines import _is_coroutine  # type: ignore[attr-defined]
from functools import _CacheInfo, _make_key, partial, partialmethod
from typing import (
def cache_invalidate(self, /, *args: Hashable, **kwargs: Any) -> bool:
    return self.__wrapper.cache_invalidate(self.__instance, *args, **kwargs)