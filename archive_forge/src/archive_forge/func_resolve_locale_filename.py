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
def resolve_locale_filename(name: os.PathLike[str] | str) -> str:
    """
    Resolve a locale identifier to a `.dat` path on disk.
    """
    name = os.path.basename(name)
    if sys.platform == 'win32' and _windows_reserved_name_re.match(os.path.splitext(name)[0]):
        raise ValueError(f'Name {name} is invalid on Windows')
    return os.path.join(_dirname, f'{name}.dat')