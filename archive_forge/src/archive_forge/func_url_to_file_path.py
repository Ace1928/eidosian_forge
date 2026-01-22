from __future__ import annotations
import hashlib
import os
from textwrap import dedent
from typing import IO, TYPE_CHECKING
from pip._vendor.cachecontrol.cache import BaseCache, SeparateBodyBaseCache
from pip._vendor.cachecontrol.controller import CacheController
def url_to_file_path(url: str, filecache: FileCache) -> str:
    """Return the file cache path based on the URL.

    This does not ensure the file exists!
    """
    key = CacheController.cache_url(url)
    return filecache._fn(key)