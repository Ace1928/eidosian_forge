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
def normalize_locale(name: str) -> str | None:
    """Normalize a locale ID by stripping spaces and apply proper casing.

    Returns the normalized locale ID string or `None` if the ID is not
    recognized.
    """
    if not name or not isinstance(name, str):
        return None
    name = name.strip().lower()
    for locale_id in chain.from_iterable([_cache, locale_identifiers()]):
        if name == locale_id.lower():
            return locale_id