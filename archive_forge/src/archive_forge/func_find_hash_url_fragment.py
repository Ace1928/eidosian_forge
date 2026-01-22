import functools
import itertools
import logging
import os
import posixpath
import re
import urllib.parse
from dataclasses import dataclass
from typing import (
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import (
from pip._internal.utils.models import KeyBasedCompareMixin
from pip._internal.utils.urls import path_to_url, url_to_path
@classmethod
@functools.lru_cache(maxsize=None)
def find_hash_url_fragment(cls, url: str) -> Optional['LinkHash']:
    """Search a string for a checksum algorithm name and encoded output value."""
    match = cls._hash_url_fragment_re.search(url)
    if match is None:
        return None
    name, value = match.groups()
    return cls(name=name, value=value)