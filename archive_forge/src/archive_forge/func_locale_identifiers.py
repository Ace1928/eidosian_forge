from __future__ import annotations
import os
import pickle
import re
import sys
import threading
from collections import abc
from collections.abc import Iterator, Mapping, MutableMapping
from functools import lru_cache
from itertools import chain
from typing import Any
@lru_cache(maxsize=None)
def locale_identifiers() -> list[str]:
    """Return a list of all locale identifiers for which locale data is
    available.

    This data is cached after the first invocation.
    You can clear the cache by calling `locale_identifiers.cache_clear()`.

    .. versionadded:: 0.8.1

    :return: a list of locale identifiers (strings)
    """
    return [stem for stem, extension in (os.path.splitext(filename) for filename in os.listdir(_dirname)) if extension == '.dat' and stem != 'root']